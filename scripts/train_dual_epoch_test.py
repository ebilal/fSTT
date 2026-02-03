import argparse
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.data import build_examples, load_dialogs, write_examples_jsonl
from src.eval import evaluate_retrieval
from src.index import VectorIndex
from src.model import build_input_examples, encode_texts
from src.utils import ensure_dir, get_device, timestamp_run_id, write_json, write_latest_pointer


def _split_examples(
    examples: List[Dict],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)
    rng.shuffle(examples)
    n = len(examples)
    val_n = int(n * val_ratio)
    test_n = int(n * test_ratio)
    train_n = max(1, n - val_n - test_n)
    train = examples[:train_n]
    val = examples[train_n:train_n + val_n]
    test = examples[train_n + val_n:]
    return train, val, test


def _eval_on_pool(model: SentenceTransformer, examples: List[Dict]) -> Dict[str, float]:
    targets = [ex["target_text"] for ex in examples]
    emb = encode_texts(model, targets)
    idx = VectorIndex.build(emb, targets, prefer_faiss=True)
    return evaluate_retrieval(model, idx, examples, topk_list=[1, 5, 10])


def main() -> None:
    p = argparse.ArgumentParser(description="Colab-friendly dual-dataset training with epoch test eval + save-best")
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--max_dialogs_multiwoz", type=int, default=8437)
    p.add_argument("--max_dialogs_dailydialog", type=int, default=11118)
    p.add_argument("--history_turns", type=int, default=6)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")

    # Tuned-ish defaults (updateable via CLI)
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--learning_rate", type=float, default=4.3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.95)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--grad_accum_steps", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--save_train_index", type=str, default="true")
    args = p.parse_args()

    device = get_device(args.device)
    save_train_index = args.save_train_index.lower() == "true"

    output_dir = args.output_dir or os.path.join(
        "outputs", timestamp_run_id(prefix="dual_run")
    )
    ensure_dir(output_dir)

    # Determinism (best effort).
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("Loading dialogs...")
    mw_train = load_dialogs("multiwoz", split="train", max_dialogs=args.max_dialogs_multiwoz)
    dd_train = load_dialogs("dailydialog", split="train", max_dialogs=args.max_dialogs_dailydialog)

    mw_ex = build_examples(mw_train, history_turns=args.history_turns)
    dd_ex = build_examples(dd_train, history_turns=args.history_turns)
    examples: List[Dict] = mw_ex + dd_ex

    train_examples, val_examples, test_examples = _split_examples(
        examples, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )

    print("Total examples:", len(examples))
    print("Train:", len(train_examples))
    print("Val:", len(val_examples))
    print("Test:", len(test_examples))

    examples_path = os.path.join(output_dir, "examples.jsonl")
    if os.path.exists(examples_path):
        os.remove(examples_path)
    write_examples_jsonl(examples_path, train_examples, split="train")
    write_examples_jsonl(examples_path, val_examples, split="val")
    write_examples_jsonl(examples_path, test_examples, split="test")

    write_json(
        os.path.join(output_dir, "metadata.json"),
        {
            "dataset": "combined",
            "sources": {
                "multiwoz_repo": "pfb30/multi_woz_v22",
                "dailydialog_repo": "roskoN/dailydialog",
            },
            "max_dialogs_multiwoz": args.max_dialogs_multiwoz,
            "max_dialogs_dailydialog": args.max_dialogs_dailydialog,
            "history_turns": args.history_turns,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "device": device,
            "train_config": {
                "model_name": args.model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "adam_betas": [args.adam_beta1, args.adam_beta2],
                "adam_eps": args.adam_eps,
                "max_grad_norm": args.max_grad_norm,
                "grad_accum_steps": args.grad_accum_steps,
            },
            "num_examples": {
                "total": len(examples),
                "train": len(train_examples),
                "val": len(val_examples),
                "test": len(test_examples),
            },
        },
    )

    model = SentenceTransformer(args.model_name, device=device)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    train_input = build_input_examples(train_examples)
    train_dl = DataLoader(
        train_input,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=model.smart_batching_collate,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    total_steps = max(
        1, (len(train_dl) * max(1, args.epochs)) // max(1, args.grad_accum_steps)
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best = {"epoch": 0, "mrr@10": float("-inf"), "metrics": {}}
    history: List[Dict] = []
    start_ts = time.time()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (features, labels) in enumerate(progress, start=1):
            features = [{k: v.to(model.device) for k, v in feat.items()} for feat in features]
            labels = labels.to(model.device)

            loss = loss_fn(features, labels)
            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            if step % max(1, args.grad_accum_steps) == 0:
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Evaluate each epoch. (We keep a val split for monitoring, but select best on test per request.)
        model.eval()
        with torch.no_grad():
            val_metrics = _eval_on_pool(model, val_examples) if val_examples else {}
            test_metrics = _eval_on_pool(model, test_examples)

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - start_ts,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)
        write_json(os.path.join(output_dir, "eval_history.json"), {"history": history})

        score = float(test_metrics.get("mrr@10", 0.0))
        if score > float(best["mrr@10"]):
            best = {"epoch": epoch, "mrr@10": score, "metrics": test_metrics}
            best_dir = os.path.join(output_dir, "encoder_best")
            model.save(best_dir)
            note_lines = [
                f"Best model updated at epoch {epoch}",
                f"test mrr@10: {score:.6f}",
                f"test recall@1: {test_metrics.get('recall@1', 0.0):.6f}",
                f"test recall@5: {test_metrics.get('recall@5', 0.0):.6f}",
                f"test recall@10: {test_metrics.get('recall@10', 0.0):.6f}",
            ]
            if val_metrics:
                note_lines += [
                    "",
                    f"val mrr@10: {val_metrics.get('mrr@10', 0.0):.6f}",
                    f"val recall@1: {val_metrics.get('recall@1', 0.0):.6f}",
                    f"val recall@5: {val_metrics.get('recall@5', 0.0):.6f}",
                    f"val recall@10: {val_metrics.get('recall@10', 0.0):.6f}",
                ]
            with open(os.path.join(output_dir, "best_note.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(note_lines) + "\n")
            write_json(os.path.join(output_dir, "best_eval.json"), best)

        # Always save an epoch checkpoint (useful in case best is earlier).
        model.save(os.path.join(output_dir, f"encoder_epoch_{epoch:02d}"))

    # Save last
    model.save(os.path.join(output_dir, "encoder_last"))

    # For compatibility with the rest of the repo (demo/index loaders expect {run}/encoder),
    # promote the best checkpoint to {run}/encoder.
    best_dir = os.path.join(output_dir, "encoder_best")
    final_encoder_dir = os.path.join(output_dir, "encoder")
    src_dir = best_dir if os.path.isdir(best_dir) else os.path.join(output_dir, "encoder_last")
    if os.path.isdir(final_encoder_dir):
        shutil.rmtree(final_encoder_dir)
    shutil.copytree(src_dir, final_encoder_dir)

    if save_train_index:
        train_targets = [ex["target_text"] for ex in train_examples]
        train_emb = encode_texts(model, train_targets)
        index = VectorIndex.build(train_emb, train_targets, prefer_faiss=True)
        index.save(os.path.join(output_dir, "index"))

    write_latest_pointer(os.path.join("outputs", "latest"), output_dir)
    print("Done. Best:", json.dumps(best, indent=2))
    print("Output dir:", output_dir)


if __name__ == "__main__":
    main()
