import argparse
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import build_examples, load_dialogs, write_examples_jsonl
from src.eval import evaluate_retrieval, sample_qualitative
from src.index import VectorIndex
from src.model import train_biencoder, encode_texts
from src.utils import ensure_dir, get_device, timestamp_run_id, write_json, write_latest_pointer


def _try_load_splits(dataset_name: str, splits, max_dialogs: int):
    last_error = None
    for split in splits:
        try:
            dialogs = load_dialogs(dataset_name, split=split, max_dialogs=max_dialogs)
            if dialogs:
                return dialogs, split
        except Exception as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ValueError(f"No dialogs found for dataset {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train listener prior bi-encoder")
    parser.add_argument("--dataset", type=str, required=True, choices=["multiwoz", "dailydialog"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_dialogs", type=int, default=2000)
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--build_index", type=str, default="true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    # Tuned defaults (see scripts/tune.py): conservative batch size, CPU-friendly.
    parser.add_argument("--learning_rate", type=float, default=4.4e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.98)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    args = parser.parse_args()

    device = get_device(args.device)
    build_index = args.build_index.lower() == "true"

    output_dir = args.output_dir or os.path.join("outputs", timestamp_run_id())
    encoder_dir = os.path.join(output_dir, "encoder")
    index_dir = os.path.join(output_dir, "index")
    ensure_dir(output_dir)

    train_dialogs, train_split = _try_load_splits(args.dataset, ["train"], args.max_dialogs)
    try:
        eval_dialogs, eval_split = _try_load_splits(
            args.dataset, ["validation", "dev", "test"], max(args.max_dialogs // 5, 1)
        )
    except Exception:
        eval_dialogs = train_dialogs[: max(args.max_dialogs // 10, 1)]
        eval_split = train_split

    train_examples = build_examples(train_dialogs, history_turns=args.history_turns)
    eval_examples = build_examples(eval_dialogs, history_turns=args.history_turns)

    examples_path = os.path.join(output_dir, "examples.jsonl")
    if os.path.exists(examples_path):
        os.remove(examples_path)
    write_examples_jsonl(examples_path, train_examples, split=train_split)
    write_examples_jsonl(examples_path, eval_examples, split=eval_split)

    encoder = train_biencoder(
        train_examples=train_examples,
        model_name=args.model_name,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=encoder_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        adam_eps=args.adam_eps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.grad_accum_steps,
    )

    metrics = {}
    qualitative = []

    if build_index:
        target_texts = [ex["target_text"] for ex in train_examples]
        embeddings = encode_texts(encoder, target_texts)
        index = VectorIndex.build(embeddings, target_texts, prefer_faiss=True)
        index.save(index_dir)

        # Evaluate on an index built from eval targets so Recall/MRR is meaningful.
        eval_targets = [ex["target_text"] for ex in eval_examples]
        eval_embeddings = encode_texts(encoder, eval_targets)
        eval_index = VectorIndex.build(eval_embeddings, eval_targets, prefer_faiss=True)

        metrics = evaluate_retrieval(encoder, eval_index, eval_examples)
        qualitative = sample_qualitative(encoder, eval_index, eval_examples, topk=args.topk)
        write_json(os.path.join(output_dir, "eval.json"), metrics)
        write_json(os.path.join(output_dir, "qualitative.json"), {"samples": qualitative})

    metadata = {
        "dataset": args.dataset,
        "train_split": train_split,
        "eval_split": eval_split,
        "max_dialogs": args.max_dialogs,
        "history_turns": args.history_turns,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": device,
        "model_name": args.model_name,
        "build_index": build_index,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "max_grad_norm": args.max_grad_norm,
        "grad_accum_steps": args.grad_accum_steps,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
    }
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    if metrics:
        print("Metrics:", json.dumps(metrics, indent=2))
    if qualitative:
        print("Qualitative samples (first 5):")
        for sample in qualitative:
            print("---")
            print(sample["history_text"])
            print("Target:", sample["target_text"])
            print("Retrieved:", sample["retrieved"])

    write_latest_pointer(os.path.join("outputs", "latest"), output_dir)
    print(f"Run saved to: {output_dir}")


if __name__ == "__main__":
    main()
