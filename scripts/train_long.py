import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data import build_examples, load_dialogs, write_examples_jsonl
from src.training import TrainConfig, train_with_time_budget
from src.utils import ensure_dir, get_device, timestamp_run_id, write_json, write_latest_pointer


def main() -> None:
    p = argparse.ArgumentParser(description="Long-running training with periodic eval + best checkpoint")
    p.add_argument("--dataset", type=str, required=True, choices=["multiwoz", "dailydialog", "combined"])
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_dialogs", type=int, default=8437)
    p.add_argument("--history_turns", type=int, default=6)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--time_budget_hours", type=float, default=8.0)
    p.add_argument("--eval_every_steps", type=int, default=2000)
    p.add_argument("--max_eval_examples", type=int, default=1000)

    # Tuned defaults (also used by scripts/train.py)
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=4.3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.95)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--grad_accum_steps", type=int, default=2)

    args = p.parse_args()

    device = get_device(args.device)
    output_dir = args.output_dir or os.path.join("outputs", timestamp_run_id(prefix="long"))
    ensure_dir(output_dir)

    if args.dataset == "combined":
        # Interpret max_dialogs as max per dataset.
        mw_train = load_dialogs("multiwoz", split="train", max_dialogs=args.max_dialogs)
        dd_train = load_dialogs("dailydialog", split="train", max_dialogs=args.max_dialogs)
        train_dialogs = mw_train + dd_train
        # Prefer validation, but fall back to a subset of train.
        try:
            mw_eval = load_dialogs("multiwoz", split="validation", max_dialogs=max(args.max_dialogs // 5, 1))
        except Exception:
            mw_eval = mw_train[: max(args.max_dialogs // 10, 1)]
        try:
            dd_eval = load_dialogs("dailydialog", split="validation", max_dialogs=max(args.max_dialogs // 5, 1))
        except Exception:
            dd_eval = dd_train[: max(args.max_dialogs // 10, 1)]
        eval_dialogs = mw_eval + dd_eval
    else:
        train_dialogs = load_dialogs(args.dataset, split="train", max_dialogs=args.max_dialogs)
        # Prefer validation, but fall back to a subset of train.
        try:
            eval_dialogs = load_dialogs(args.dataset, split="validation", max_dialogs=max(args.max_dialogs // 5, 1))
        except Exception:
            eval_dialogs = train_dialogs[: max(args.max_dialogs // 10, 1)]

    train_examples = build_examples(train_dialogs, history_turns=args.history_turns)
    eval_examples = build_examples(eval_dialogs, history_turns=args.history_turns)

    examples_path = os.path.join(output_dir, "examples.jsonl")
    if os.path.exists(examples_path):
        os.remove(examples_path)
    write_examples_jsonl(examples_path, train_examples, split="train")
    write_examples_jsonl(examples_path, eval_examples, split="eval")

    cfg = TrainConfig(
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_betas=(args.adam_beta1, args.adam_beta2),
        adam_eps=args.adam_eps,
        max_grad_norm=args.max_grad_norm,
        grad_accum_steps=args.grad_accum_steps,
    )

    write_json(
        os.path.join(output_dir, "metadata.json"),
        {
            "dataset": args.dataset,
            "max_dialogs": args.max_dialogs,
            "history_turns": args.history_turns,
            "device": device,
            "time_budget_hours": args.time_budget_hours,
            "eval_every_steps": args.eval_every_steps,
            "max_eval_examples": args.max_eval_examples,
            "train_config": {
                "model_name": cfg.model_name,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "warmup_ratio": cfg.warmup_ratio,
                "weight_decay": cfg.weight_decay,
                "adam_betas": list(cfg.adam_betas),
                "adam_eps": cfg.adam_eps,
                "max_grad_norm": cfg.max_grad_norm,
                "grad_accum_steps": cfg.grad_accum_steps,
            },
            "num_train_examples": len(train_examples),
            "num_eval_examples": len(eval_examples),
        },
    )

    summary = train_with_time_budget(
        train_examples=train_examples,
        eval_examples=eval_examples,
        output_dir=output_dir,
        cfg=cfg,
        time_budget_hours=args.time_budget_hours,
        eval_every_steps=args.eval_every_steps,
        max_eval_examples=args.max_eval_examples,
    )

    # Convenience pointers
    write_latest_pointer(os.path.join("outputs", "latest_long"), output_dir)
    write_latest_pointer(os.path.join("outputs", "latest"), output_dir)

    print("Training complete.")
    print(summary)


if __name__ == "__main__":
    main()
