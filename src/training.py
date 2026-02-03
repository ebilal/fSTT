import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .eval import evaluate_retrieval
from .index import VectorIndex
from .model import build_input_examples, encode_texts
from .utils import ensure_dir, write_json


@dataclass
class TrainConfig:
    model_name: str
    device: str
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    adam_betas: Tuple[float, float]
    adam_eps: float
    max_grad_norm: float
    grad_accum_steps: int


class BestCheckpoint:
    def __init__(self) -> None:
        self.best_score = float("-inf")
        self.best_metrics: Dict[str, float] = {}
        self.best_step = 0


def train_with_time_budget(
    train_examples: List[dict],
    eval_examples: List[dict],
    output_dir: str,
    cfg: TrainConfig,
    time_budget_hours: float,
    eval_every_steps: int = 2000,
    eval_k: int = 10,
    max_eval_examples: int = 1000,
) -> Dict[str, object]:
    """Train until wall-clock budget is exhausted, evaluating periodically.

    Saves:
      - {output_dir}/encoder_last
      - {output_dir}/encoder_best (whenever improved)
      - {output_dir}/best_eval.json
      - {output_dir}/train_progress.json

    Returns a dict summary.
    """

    ensure_dir(output_dir)

    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    train_input = build_input_examples(train_examples)
    train_dl = DataLoader(
        train_input,
        shuffle=True,
        batch_size=cfg.batch_size,
        collate_fn=model.smart_batching_collate,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    # Approximate schedule length: treat one full pass over dataloader as an "epoch" and
    # extend linearly with time; we set a large num_training_steps and stop by time anyway.
    # This avoids re-creating the scheduler mid-run.
    approx_total_steps = max(1, (len(train_dl) * 100) // max(1, cfg.grad_accum_steps))
    warmup_steps = int(approx_total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=approx_total_steps,
    )

    if max_eval_examples and len(eval_examples) > max_eval_examples:
        eval_examples = eval_examples[:max_eval_examples]

    best = BestCheckpoint()
    start = time.time()
    deadline = start + (time_budget_hours * 3600.0)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    def _evaluate_and_maybe_save(step: int) -> None:
        # Build an eval-target pool so MRR/Recall is meaningful.
        eval_targets = [ex["target_text"] for ex in eval_examples]
        eval_target_emb = encode_texts(model, eval_targets)
        eval_index = VectorIndex.build(eval_target_emb, eval_targets, prefer_faiss=True)
        metrics = evaluate_retrieval(model, eval_index, eval_examples, topk_list=[1, 5, 10])

        score = float(metrics.get("mrr@10", 0.0))
        progress = {
            "step": step,
            "elapsed_sec": time.time() - start,
            "metrics": metrics,
            "score_mrr@10": score,
        }
        write_json(os.path.join(output_dir, "train_progress.json"), progress)

        if score > best.best_score:
            best.best_score = score
            best.best_metrics = metrics
            best.best_step = step
            best_dir = os.path.join(output_dir, "encoder_best")
            model.save(best_dir)
            write_json(
                os.path.join(output_dir, "best_eval.json"),
                {
                    "best_step": best.best_step,
                    "best_score_mrr@10": best.best_score,
                    "best_metrics": best.best_metrics,
                },
            )

    # Evaluate once at start (mostly for baseline), then train.
    _evaluate_and_maybe_save(step=0)

    # Loop until deadline, cycling over dataloader.
    while time.time() < deadline:
        progress = tqdm(train_dl, desc="Training", leave=False)
        for features, labels in progress:
            if time.time() >= deadline:
                break

            global_step += 1
            features = [{k: v.to(model.device) for k, v in feat.items()} for feat in features]
            labels = labels.to(model.device)

            loss = loss_fn(features, labels)
            loss = loss / max(1, cfg.grad_accum_steps)
            loss.backward()

            if global_step % max(1, cfg.grad_accum_steps) == 0:
                if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            if eval_every_steps and (global_step % eval_every_steps == 0):
                _evaluate_and_maybe_save(step=global_step)

    # Final eval + save last.
    _evaluate_and_maybe_save(step=global_step)
    last_dir = os.path.join(output_dir, "encoder_last")
    model.save(last_dir)

    summary = {
        "trained_steps": global_step,
        "time_budget_hours": time_budget_hours,
        "best_step": best.best_step,
        "best_score_mrr@10": best.best_score,
        "best_metrics": best.best_metrics,
        "encoder_best": os.path.join(output_dir, "encoder_best"),
        "encoder_last": last_dir,
    }
    write_json(os.path.join(output_dir, "train_summary.json"), summary)
    return summary
