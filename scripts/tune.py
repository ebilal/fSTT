import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import build_examples, load_dialogs, write_examples_jsonl
from src.eval import evaluate_retrieval
from src.index import VectorIndex
from src.model import train_biencoder, encode_texts
from src.utils import ensure_dir, get_device, timestamp_run_id, write_json, write_latest_pointer


@dataclass
class TrialConfig:
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    max_grad_norm: float
    grad_accum_steps: int


def _try_load_split(dataset_name: str, split: str, max_dialogs: int):
    dialogs = load_dialogs(dataset_name, split=split, max_dialogs=max_dialogs)
    if not dialogs:
        raise ValueError(f"No dialogs found for dataset {dataset_name} split {split}")
    return dialogs


def sample_trial(rng: random.Random) -> TrialConfig:
    batch_size = rng.choice([16, 32])
    learning_rate = 10 ** rng.uniform(-5.0, -4.2)  # ~1e-5..6e-5
    warmup_ratio = rng.choice([0.0, 0.03, 0.06, 0.1])
    weight_decay = rng.choice([0.0, 0.01, 0.05])
    adam_beta1 = rng.choice([0.9, 0.95])
    adam_beta2 = rng.choice([0.999, 0.98])
    adam_eps = rng.choice([1e-8, 1e-6])
    max_grad_norm = rng.choice([0.0, 1.0])
    grad_accum_steps = rng.choice([1, 2])
    return TrialConfig(
        batch_size=batch_size,
        learning_rate=float(learning_rate),
        warmup_ratio=float(warmup_ratio),
        weight_decay=float(weight_decay),
        adam_beta1=float(adam_beta1),
        adam_beta2=float(adam_beta2),
        adam_eps=float(adam_eps),
        max_grad_norm=float(max_grad_norm),
        grad_accum_steps=int(grad_accum_steps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for listener prior bi-encoder")
    parser.add_argument("--dataset", type=str, required=True, choices=["multiwoz", "dailydialog", "combined"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_dialogs", type=int, default=500, help="Used for single-dataset tuning.")
    parser.add_argument("--max_dialogs_multiwoz", type=int, default=800, help="Used when --dataset combined.")
    parser.add_argument("--max_dialogs_dailydialog", type=int, default=1000, help="Used when --dataset combined.")
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_val_examples", type=int, default=2000)
    parser.add_argument("--max_test_examples", type=int, default=4000)
    args = parser.parse_args()

    device = get_device(args.device)
    rng = random.Random(args.seed)

    output_dir = args.output_dir or os.path.join("outputs", timestamp_run_id(prefix="tune"))
    ensure_dir(output_dir)

    if args.dataset == "combined":
        # Build combined examples, then split into train/val/test.
        mw_dialogs = _try_load_split("multiwoz", "train", args.max_dialogs_multiwoz)
        dd_dialogs = _try_load_split("dailydialog", "train", args.max_dialogs_dailydialog)

        mw_ex = build_examples(mw_dialogs, history_turns=args.history_turns)
        dd_ex = build_examples(dd_dialogs, history_turns=args.history_turns)
        all_examples = mw_ex + dd_ex
        rng.shuffle(all_examples)

        n = len(all_examples)
        val_n = int(n * args.val_ratio)
        test_n = int(n * args.test_ratio)
        train_n = n - val_n - test_n
        if train_n <= 0 or val_n <= 0 or test_n <= 0:
            raise ValueError(f"Invalid split sizes: n={n}, train={train_n}, val={val_n}, test={test_n}")

        train_examples = all_examples[:train_n]
        val_examples = all_examples[train_n : train_n + val_n]
        test_examples = all_examples[train_n + val_n :]
    else:
        train_dialogs = _try_load_split(args.dataset, "train", args.max_dialogs)
        try:
            eval_dialogs = _try_load_split(args.dataset, "validation", max(args.max_dialogs // 5, 1))
        except Exception:
            eval_dialogs = train_dialogs[: max(args.max_dialogs // 10, 1)]

        train_examples = build_examples(train_dialogs, history_turns=args.history_turns)
        val_examples = build_examples(eval_dialogs, history_turns=args.history_turns)
        test_examples = val_examples

    if args.max_val_examples and len(val_examples) > args.max_val_examples:
        val_examples = val_examples[: args.max_val_examples]
    if args.max_test_examples and len(test_examples) > args.max_test_examples:
        test_examples = test_examples[: args.max_test_examples]

    if len(train_examples) < 200 or len(val_examples) < 50 or len(test_examples) < 50:
        raise ValueError(
            f"Not enough examples for tuning (train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)})"
        )

    examples_path = os.path.join(output_dir, "examples.jsonl")
    if os.path.exists(examples_path):
        os.remove(examples_path)
    write_examples_jsonl(examples_path, train_examples, split="train")
    write_examples_jsonl(examples_path, val_examples, split="val")
    write_examples_jsonl(examples_path, test_examples, split="test")

    best_score = -1.0
    best_trial = None
    results: List[Dict] = []

    for t in range(1, args.trials + 1):
        trial_cfg = sample_trial(rng)
        trial_id = f"trial_{t:03d}"
        trial_dir = os.path.join(output_dir, trial_id)
        encoder_dir = os.path.join(trial_dir, "encoder")
        index_dir = os.path.join(trial_dir, "index")
        ensure_dir(trial_dir)

        encoder = train_biencoder(
            train_examples=train_examples,
            model_name=args.model_name,
            device=device,
            epochs=args.epochs,
            batch_size=trial_cfg.batch_size,
            output_dir=encoder_dir,
            learning_rate=trial_cfg.learning_rate,
            warmup_ratio=trial_cfg.warmup_ratio,
            weight_decay=trial_cfg.weight_decay,
            adam_betas=(trial_cfg.adam_beta1, trial_cfg.adam_beta2),
            adam_eps=trial_cfg.adam_eps,
            max_grad_norm=trial_cfg.max_grad_norm,
            gradient_accumulation_steps=trial_cfg.grad_accum_steps,
        )

        # Save an index for inspection (mainly for qualitative); selection uses val/test pool indices.
        train_targets = [ex["target_text"] for ex in train_examples]
        train_emb = encode_texts(encoder, train_targets)
        train_index = VectorIndex.build(train_emb, train_targets, prefer_faiss=True)
        train_index.save(index_dir)

        val_targets = [ex["target_text"] for ex in val_examples]
        val_emb = encode_texts(encoder, val_targets)
        val_index = VectorIndex.build(val_emb, val_targets, prefer_faiss=True)
        val_metrics = evaluate_retrieval(encoder, val_index, val_examples)

        test_targets = [ex["target_text"] for ex in test_examples]
        test_emb = encode_texts(encoder, test_targets)
        test_index = VectorIndex.build(test_emb, test_targets, prefer_faiss=True)
        test_metrics = evaluate_retrieval(encoder, test_index, test_examples)

        score = float(val_metrics.get("mrr@10", 0.0))

        record = {
            "trial": trial_id,
            "score_mrr@10_val": score,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "config": asdict(trial_cfg),
        }
        results.append(record)
        write_json(os.path.join(trial_dir, "trial.json"), record)

        if score > best_score:
            best_score = score
            best_trial = record
            write_latest_pointer(os.path.join(output_dir, "best"), trial_dir)
            write_json(os.path.join(output_dir, "best_trial.json"), best_trial)
            print(f"New best: {trial_id} val_mrr@10={best_score:.4f} (test_mrr@10={test_metrics.get('mrr@10', 0.0):.4f})")

    write_json(os.path.join(output_dir, "all_trials.json"), {"results": results})
    if best_trial is None:
        raise RuntimeError("No successful trials")

    write_latest_pointer(os.path.join("outputs", "latest_tune"), output_dir)
    print(f"Tuning finished. Best val_mrr@10={best_score:.4f}")
    print(f"Best trial dir: {os.path.join(output_dir, best_trial['trial'])}")


if __name__ == "__main__":
    main()
