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
from src.prior import extract_priors
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


def build_keyterm_examples(
    dialogs: List[List[Tuple[str, str]]],
    history_turns: int,
    target_role: str = "SYSTEM",
    max_keywords: int = 30,
    max_keyterms: int = 30,
) -> List[Dict]:
    """Build examples where target is keyterms/keywords instead of full text."""
    from src.data import _normalize_role
    
    target_role = _normalize_role(target_role)
    examples: List[Dict] = []
    
    for dialog in dialogs:
        for idx, (role, text) in enumerate(dialog):
            if _normalize_role(role) != target_role:
                continue
            start = max(0, idx - history_turns)
            history = dialog[start:idx]
            if not history:
                continue
            history_text = "\n".join([f"{r}: {t}" for r, t in history])
            
            # Extract keyterms from the target utterance
            keyterms_dict = extract_priors([text], max_keywords=max_keywords, max_keyterms=max_keyterms)
            
            # Combine keywords and keyterms into a single target string for training
            # This is what the model will learn to predict
            all_keyterms = keyterms_dict["keywords"] + keyterms_dict["keyterms"]
            target_keyterms_text = ", ".join(all_keyterms) if all_keyterms else ""
            
            examples.append({
                "history_text": history_text,
                "target_text": target_keyterms_text,  # Use keyterms as target for training
                "target_original_text": text,  # Keep original for reference/evaluation
                "target_keywords": keyterms_dict["keywords"],
                "target_keyterms": keyterms_dict["keyterms"],
            })
    return examples


def _eval_keyterm_retrieval(
    model: SentenceTransformer,
    examples: List[Dict],
    topk: int = 20,
) -> Dict[str, float]:
    """Evaluate keyterm prediction by retrieving top-k keyterm candidates and checking overlap."""
    if not examples:
        return {
            "keyterm_precision@5": 0.0, "keyterm_recall@5": 0.0,
            "keyterm_precision@10": 0.0, "keyterm_recall@10": 0.0,
            "keyterm_precision@20": 0.0, "keyterm_recall@20": 0.0,
            "keyword_precision@5": 0.0, "keyword_recall@5": 0.0,
            "keyword_precision@10": 0.0, "keyword_recall@10": 0.0,
            "keyword_precision@20": 0.0, "keyword_recall@20": 0.0,
        }
    
    # Build index from all target keyterm strings (what model was trained to predict)
    target_keyterm_texts = [ex["target_text"] for ex in examples]  # These are keyterm strings
    target_emb = encode_texts(model, target_keyterm_texts)
    index = VectorIndex.build(target_emb, target_keyterm_texts, prefer_faiss=True)
    
    histories = [ex["history_text"] for ex in examples]
    query_emb = encode_texts(model, histories)
    indices, _ = index.retrieve(query_emb, topk)
    
    keyword_precisions_5 = []
    keyword_recalls_5 = []
    keyterm_precisions_5 = []
    keyterm_recalls_5 = []
    keyword_precisions_10 = []
    keyword_recalls_10 = []
    keyterm_precisions_10 = []
    keyterm_recalls_10 = []
    keyword_precisions_20 = []
    keyword_recalls_20 = []
    keyterm_precisions_20 = []
    keyterm_recalls_20 = []
    
    for i, row in enumerate(indices):
        # Retrieved keyterm strings (comma-separated) - top 20
        retrieved_keyterm_strings = [index.target_texts[idx] for idx in row]
        
        # For @5: use first 5 retrieved strings
        retrieved_keyterm_strings_5 = retrieved_keyterm_strings[:5]
        # For @10: use first 10 retrieved strings
        retrieved_keyterm_strings_10 = retrieved_keyterm_strings[:10]
        
        # Parse retrieved keyterms back into lists by splitting on commas
        # Collect all terms from top-20 retrieved keyterm strings
        all_retrieved_terms_20 = []
        for kt_string in retrieved_keyterm_strings:
            if kt_string:  # Handle empty strings
                terms = [t.strip() for t in kt_string.split(",") if t.strip()]
                all_retrieved_terms_20.extend(terms)
        
        # Collect all terms from top-10 retrieved keyterm strings
        all_retrieved_terms_10 = []
        for kt_string in retrieved_keyterm_strings_10:
            if kt_string:  # Handle empty strings
                terms = [t.strip() for t in kt_string.split(",") if t.strip()]
                all_retrieved_terms_10.extend(terms)
        
        # Collect all terms from top-5 retrieved keyterm strings
        all_retrieved_terms_5 = []
        for kt_string in retrieved_keyterm_strings_5:
            if kt_string:  # Handle empty strings
                terms = [t.strip() for t in kt_string.split(",") if t.strip()]
                all_retrieved_terms_5.extend(terms)
        
        # Classify as keyword (single word) or keyterm (multi-word) for @20
        pred_keywords_20 = set()
        pred_keyterms_20 = set()
        for term in all_retrieved_terms_20:
            words = term.split()
            if len(words) == 1:
                pred_keywords_20.add(term.lower())
            elif len(words) >= 2:  # Multi-word phrases are keyterms
                pred_keyterms_20.add(term.lower())
        
        # Classify as keyword (single word) or keyterm (multi-word) for @10
        pred_keywords_10 = set()
        pred_keyterms_10 = set()
        for term in all_retrieved_terms_10:
            words = term.split()
            if len(words) == 1:
                pred_keywords_10.add(term.lower())
            elif len(words) >= 2:  # Multi-word phrases are keyterms
                pred_keyterms_10.add(term.lower())
        
        # Classify as keyword (single word) or keyterm (multi-word) for @5
        pred_keywords_5 = set()
        pred_keyterms_5 = set()
        for term in all_retrieved_terms_5:
            words = term.split()
            if len(words) == 1:
                pred_keywords_5.add(term.lower())
            elif len(words) >= 2:  # Multi-word phrases are keyterms
                pred_keyterms_5.add(term.lower())
        
        # Ground truth keyterms (normalized to lowercase for comparison)
        gt_keywords = set(k.lower() for k in examples[i]["target_keywords"])
        gt_keyterms = set(k.lower() for k in examples[i]["target_keyterms"])
        
        # @5 metrics
        if pred_keywords_5:
            keyword_prec_5 = len(gt_keywords & pred_keywords_5) / len(pred_keywords_5)
            keyword_precisions_5.append(keyword_prec_5)
        else:
            keyword_precisions_5.append(0.0)
        
        if gt_keywords:
            keyword_rec_5 = len(gt_keywords & pred_keywords_5) / len(gt_keywords)
            keyword_recalls_5.append(keyword_rec_5)
        else:
            keyword_recalls_5.append(0.0)
        
        if pred_keyterms_5:
            keyterm_prec_5 = len(gt_keyterms & pred_keyterms_5) / len(pred_keyterms_5)
            keyterm_precisions_5.append(keyterm_prec_5)
        else:
            keyterm_precisions_5.append(0.0)
        
        if gt_keyterms:
            keyterm_rec_5 = len(gt_keyterms & pred_keyterms_5) / len(gt_keyterms)
            keyterm_recalls_5.append(keyterm_rec_5)
        else:
            keyterm_recalls_5.append(0.0)
        
        # @10 metrics
        if pred_keywords_10:
            keyword_prec_10 = len(gt_keywords & pred_keywords_10) / len(pred_keywords_10)
            keyword_precisions_10.append(keyword_prec_10)
        else:
            keyword_precisions_10.append(0.0)
        
        if gt_keywords:
            keyword_rec_10 = len(gt_keywords & pred_keywords_10) / len(gt_keywords)
            keyword_recalls_10.append(keyword_rec_10)
        else:
            keyword_recalls_10.append(0.0)
        
        if pred_keyterms_10:
            keyterm_prec_10 = len(gt_keyterms & pred_keyterms_10) / len(pred_keyterms_10)
            keyterm_precisions_10.append(keyterm_prec_10)
        else:
            keyterm_precisions_10.append(0.0)
        
        if gt_keyterms:
            keyterm_rec_10 = len(gt_keyterms & pred_keyterms_10) / len(gt_keyterms)
            keyterm_recalls_10.append(keyterm_rec_10)
        else:
            keyterm_recalls_10.append(0.0)
        
        # @20 metrics
        if pred_keywords_20:
            keyword_prec_20 = len(gt_keywords & pred_keywords_20) / len(pred_keywords_20)
            keyword_precisions_20.append(keyword_prec_20)
        else:
            keyword_precisions_20.append(0.0)
        
        if gt_keywords:
            keyword_rec_20 = len(gt_keywords & pred_keywords_20) / len(gt_keywords)
            keyword_recalls_20.append(keyword_rec_20)
        else:
            keyword_recalls_20.append(0.0)
        
        if pred_keyterms_20:
            keyterm_prec_20 = len(gt_keyterms & pred_keyterms_20) / len(pred_keyterms_20)
            keyterm_precisions_20.append(keyterm_prec_20)
        else:
            keyterm_precisions_20.append(0.0)
        
        if gt_keyterms:
            keyterm_rec_20 = len(gt_keyterms & pred_keyterms_20) / len(gt_keyterms)
            keyterm_recalls_20.append(keyterm_rec_20)
        else:
            keyterm_recalls_20.append(0.0)
    
    return {
        "keyword_precision@5": sum(keyword_precisions_5) / len(keyword_precisions_5) if keyword_precisions_5 else 0.0,
        "keyword_recall@5": sum(keyword_recalls_5) / len(keyword_recalls_5) if keyword_recalls_5 else 0.0,
        "keyterm_precision@5": sum(keyterm_precisions_5) / len(keyterm_precisions_5) if keyterm_precisions_5 else 0.0,
        "keyterm_recall@5": sum(keyterm_recalls_5) / len(keyterm_recalls_5) if keyterm_recalls_5 else 0.0,
        "keyword_precision@10": sum(keyword_precisions_10) / len(keyword_precisions_10) if keyword_precisions_10 else 0.0,
        "keyword_recall@10": sum(keyword_recalls_10) / len(keyword_recalls_10) if keyword_recalls_10 else 0.0,
        "keyterm_precision@10": sum(keyterm_precisions_10) / len(keyterm_precisions_10) if keyterm_precisions_10 else 0.0,
        "keyterm_recall@10": sum(keyterm_recalls_10) / len(keyterm_recalls_10) if keyterm_recalls_10 else 0.0,
        "keyword_precision@20": sum(keyword_precisions_20) / len(keyword_precisions_20) if keyword_precisions_20 else 0.0,
        "keyword_recall@20": sum(keyword_recalls_20) / len(keyword_recalls_20) if keyword_recalls_20 else 0.0,
        "keyterm_precision@20": sum(keyterm_precisions_20) / len(keyterm_precisions_20) if keyterm_precisions_20 else 0.0,
        "keyterm_recall@20": sum(keyterm_recalls_20) / len(keyterm_recalls_20) if keyterm_recalls_20 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Colab-friendly dual-dataset training optimized for keyterm prediction")
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
    p.add_argument("--target_role", type=str, default="SYSTEM")
    p.add_argument("--max_keywords", type=int, default=30)
    p.add_argument("--max_keyterms", type=int, default=30)
    p.add_argument("--save_train_index", type=str, default="true")
    args = p.parse_args()

    device = get_device(args.device)
    save_train_index = args.save_train_index.lower() == "true"

    output_dir = args.output_dir or os.path.join(
        "outputs", timestamp_run_id(prefix="keyterms_run")
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

    print("Building keyterm examples...")
    mw_ex = build_keyterm_examples(mw_train, history_turns=args.history_turns, target_role=args.target_role, 
                                   max_keywords=args.max_keywords, max_keyterms=args.max_keyterms)
    dd_ex = build_keyterm_examples(dd_train, history_turns=args.history_turns, target_role=args.target_role,
                                   max_keywords=args.max_keywords, max_keyterms=args.max_keyterms)
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
            "optimization_goal": "keyterm_prediction",
            "sources": {
                "multiwoz_repo": "pfb30/multi_woz_v22",
                "dailydialog_repo": "roskoN/dailydialog",
            },
            "max_dialogs_multiwoz": args.max_dialogs_multiwoz,
            "max_dialogs_dailydialog": args.max_dialogs_dailydialog,
            "history_turns": args.history_turns,
            "target_role": args.target_role,
            "max_keywords": args.max_keywords,
            "max_keyterms": args.max_keyterms,
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

    # Train directly on keyterm targets - model learns to predict keyterms from history
    model = SentenceTransformer(args.model_name, device=device)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    # Build examples for training - target is keyterms, not full text
    # This directly optimizes the model to predict keyterms
    train_input = build_input_examples(train_examples)  # Uses target_text which is now keyterms
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

    best = {"epoch": 0, "keyword_recall@20": float("-inf"), "metrics": {}}
    history: List[Dict] = []
    start_ts = time.time()

    # Initial evaluation (baseline before training)
    print("\n" + "="*60)
    print("Initial Evaluation (Before Training):")
    print("="*60)
    model.eval()
    with torch.no_grad():
        val_metrics = _eval_keyterm_retrieval(model, val_examples) if val_examples else {}
        test_metrics = _eval_keyterm_retrieval(model, test_examples)
    test_keyterm_f1_5 = 0.0
    if test_metrics.get('keyterm_precision@5', 0.0) + test_metrics.get('keyterm_recall@5', 0.0) > 0:
        test_keyterm_f1_5 = (2 * test_metrics.get('keyterm_precision@5', 0.0) * test_metrics.get('keyterm_recall@5', 0.0)) / \
                           (test_metrics.get('keyterm_precision@5', 0.0) + test_metrics.get('keyterm_recall@5', 0.0))
    test_keyterm_f1_10 = 0.0
    if test_metrics.get('keyterm_precision@10', 0.0) + test_metrics.get('keyterm_recall@10', 0.0) > 0:
        test_keyterm_f1_10 = (2 * test_metrics.get('keyterm_precision@10', 0.0) * test_metrics.get('keyterm_recall@10', 0.0)) / \
                            (test_metrics.get('keyterm_precision@10', 0.0) + test_metrics.get('keyterm_recall@10', 0.0))
    
    print(f"  Test Set - Keyterm Precision@5:  {test_metrics.get('keyterm_precision@5', 0.0):.6f}, "
          f"Keyterm Recall@5:  {test_metrics.get('keyterm_recall@5', 0.0):.6f}, "
          f"Keyterm F1@5:  {test_keyterm_f1_5:.6f}")
    print(f"  Test Set - Keyterm Precision@10: {test_metrics.get('keyterm_precision@10', 0.0):.6f}, "
          f"Keyterm Recall@10: {test_metrics.get('keyterm_recall@10', 0.0):.6f}, "
          f"Keyterm F1@10: {test_keyterm_f1_10:.6f}")
    print(f"  Test Set - Keyword Precision@10: {test_metrics.get('keyword_precision@10', 0.0):.6f}, "
          f"Keyword Recall@10: {test_metrics.get('keyword_recall@10', 0.0):.6f}")
    print(f"  Test Set - Keyword Precision@20: {test_metrics.get('keyword_precision@20', 0.0):.6f}, "
          f"Keyword Recall@20: {test_metrics.get('keyword_recall@20', 0.0):.6f} ⭐ (selection metric)")
    if val_metrics:
        val_keyterm_f1_5 = 0.0
        if val_metrics.get('keyterm_precision@5', 0.0) + val_metrics.get('keyterm_recall@5', 0.0) > 0:
            val_keyterm_f1_5 = (2 * val_metrics.get('keyterm_precision@5', 0.0) * val_metrics.get('keyterm_recall@5', 0.0)) / \
                              (val_metrics.get('keyterm_precision@5', 0.0) + val_metrics.get('keyterm_recall@5', 0.0))
        val_keyterm_f1_10 = 0.0
        if val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0) > 0:
            val_keyterm_f1_10 = (2 * val_metrics.get('keyterm_precision@10', 0.0) * val_metrics.get('keyterm_recall@10', 0.0)) / \
                               (val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0))
        print(f"  Val Set   - Keyterm F1@5:  {val_keyterm_f1_5:.6f}, "
              f"Keyterm F1@10: {val_keyterm_f1_10:.6f}")
    print("="*60 + "\n")

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

        # Evaluate each epoch on keyterm metrics
        model.eval()
        with torch.no_grad():
            val_metrics = _eval_keyterm_retrieval(model, val_examples) if val_examples else {}
            test_metrics = _eval_keyterm_retrieval(model, test_examples)

        # Calculate F1 scores for @5 and @10
        keyterm_f1_5 = 0.0
        if test_metrics.get('keyterm_precision@5', 0.0) + test_metrics.get('keyterm_recall@5', 0.0) > 0:
            keyterm_f1_5 = (2 * test_metrics.get('keyterm_precision@5', 0.0) * test_metrics.get('keyterm_recall@5', 0.0)) / \
                          (test_metrics.get('keyterm_precision@5', 0.0) + test_metrics.get('keyterm_recall@5', 0.0))
        
        keyterm_f1_10 = 0.0
        if test_metrics.get('keyterm_precision@10', 0.0) + test_metrics.get('keyterm_recall@10', 0.0) > 0:
            keyterm_f1_10 = (2 * test_metrics.get('keyterm_precision@10', 0.0) * test_metrics.get('keyterm_recall@10', 0.0)) / \
                           (test_metrics.get('keyterm_precision@10', 0.0) + test_metrics.get('keyterm_recall@10', 0.0))
        
        keyword_f1_10 = 0.0
        if test_metrics.get('keyword_precision@10', 0.0) + test_metrics.get('keyword_recall@10', 0.0) > 0:
            keyword_f1_10 = (2 * test_metrics.get('keyword_precision@10', 0.0) * test_metrics.get('keyword_recall@10', 0.0)) / \
                           (test_metrics.get('keyword_precision@10', 0.0) + test_metrics.get('keyword_recall@10', 0.0))
        
        test_metrics["keyterm_f1@5"] = keyterm_f1_5
        test_metrics["keyterm_f1@10"] = keyterm_f1_10
        test_metrics["keyword_f1@10"] = keyword_f1_10

        # Print performance metrics on test set
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} - Keyterm Prediction Performance (Test Set):")
        print(f"{'='*60}")
        print(f"  Keyterm Precision@5:  {test_metrics.get('keyterm_precision@5', 0.0):.6f}")
        print(f"  Keyterm Recall@5:     {test_metrics.get('keyterm_recall@5', 0.0):.6f}")
        print(f"  Keyterm F1@5:         {keyterm_f1_5:.6f}")
        print(f"  Keyterm Precision@10: {test_metrics.get('keyterm_precision@10', 0.0):.6f}")
        print(f"  Keyterm Recall@10:    {test_metrics.get('keyterm_recall@10', 0.0):.6f}")
        print(f"  Keyterm F1@10:        {keyterm_f1_10:.6f}")
        print(f"  Keyterm Precision@20: {test_metrics.get('keyterm_precision@20', 0.0):.6f}")
        print(f"  Keyterm Recall@20:    {test_metrics.get('keyterm_recall@20', 0.0):.6f}")
        print(f"  Keyword Precision@10: {test_metrics.get('keyword_precision@10', 0.0):.6f}")
        print(f"  Keyword Recall@10:   {test_metrics.get('keyword_recall@10', 0.0):.6f}")
        print(f"  Keyword F1@10:        {keyword_f1_10:.6f}")
        print(f"  Keyword Precision@20: {test_metrics.get('keyword_precision@20', 0.0):.6f}")
        print(f"  Keyword Recall@20:   {test_metrics.get('keyword_recall@20', 0.0):.6f} ⭐ (selection metric)")
        if val_metrics:
            print(f"\n  Validation Set Performance:")
            val_keyterm_f1_5 = 0.0
            if val_metrics.get('keyterm_precision@5', 0.0) + val_metrics.get('keyterm_recall@5', 0.0) > 0:
                val_keyterm_f1_5 = (2 * val_metrics.get('keyterm_precision@5', 0.0) * val_metrics.get('keyterm_recall@5', 0.0)) / \
                                  (val_metrics.get('keyterm_precision@5', 0.0) + val_metrics.get('keyterm_recall@5', 0.0))
            val_keyterm_f1_10 = 0.0
            if val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0) > 0:
                val_keyterm_f1_10 = (2 * val_metrics.get('keyterm_precision@10', 0.0) * val_metrics.get('keyterm_recall@10', 0.0)) / \
                                   (val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0))
            print(f"    Keyterm F1@5:         {val_keyterm_f1_5:.6f}")
            print(f"    Keyterm F1@10:        {val_keyterm_f1_10:.6f}")
            print(f"    Keyword F1@10:        {val_metrics.get('keyword_f1@10', 0.0):.6f}")
        print(f"{'='*60}\n")

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - start_ts,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(record)
        write_json(os.path.join(output_dir, "eval_history.json"), {"history": history})

        score = test_metrics.get('keyword_recall@20', 0.0)  # Use Keyword Recall@20 for model selection
        best_score = float(best["keyword_recall@20"])
        if score > best_score:
            print(f"🎉 NEW BEST MODEL! (Keyword Recall@20 improved from {best_score:.6f} to {score:.6f})")
            best = {
                "epoch": epoch,
                "keyterm_f1@5": keyterm_f1_5,
                "keyterm_f1@10": keyterm_f1_10,
                "keyword_recall@20": score,
                "metrics": test_metrics,
                "model_name": args.model_name,
                "model_type": "SentenceTransformer",
                "base_model": args.model_name,
                "optimization_goal": "keyterm_prediction",
                "training_config": {
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "adam_betas": [args.adam_beta1, args.adam_beta2],
                    "adam_eps": args.adam_eps,
                    "grad_accum_steps": args.grad_accum_steps,
                    "warmup_ratio": args.warmup_ratio,
                    "max_grad_norm": args.max_grad_norm,
                },
                "dataset_info": {
                    "target_role": args.target_role,
                    "history_turns": args.history_turns,
                    "max_keywords": args.max_keywords,
                    "max_keyterms": args.max_keyterms,
                    "num_train_examples": len(train_examples),
                    "num_val_examples": len(val_examples),
                    "num_test_examples": len(test_examples),
                },
            }
            best_dir = os.path.join(output_dir, "encoder_best")
            model.save(best_dir)
            print(f"💾 Saved best model to: {best_dir}")
            
            note_lines = [
                f"Best model updated at epoch {epoch}",
                f"",
                f"Model Information:",
                f"  Model Type: SentenceTransformer",
                f"  Base Model: {args.model_name}",
                f"  Optimization Goal: Keyterm Prediction",
                f"",
                f"Test Set Performance (Keyterm Prediction):",
                f"  Keyterm Precision@5:  {test_metrics.get('keyterm_precision@5', 0.0):.6f}",
                f"  Keyterm Recall@5:     {test_metrics.get('keyterm_recall@5', 0.0):.6f}",
                f"  Keyterm F1@5:         {keyterm_f1_5:.6f}",
                f"  Keyterm Precision@10: {test_metrics.get('keyterm_precision@10', 0.0):.6f}",
                f"  Keyterm Recall@10:    {test_metrics.get('keyterm_recall@10', 0.0):.6f}",
                f"  Keyterm F1@10:        {keyterm_f1_10:.6f}",
                f"  Keyterm Precision@20: {test_metrics.get('keyterm_precision@20', 0.0):.6f}",
                f"  Keyterm Recall@20:    {test_metrics.get('keyterm_recall@20', 0.0):.6f}",
                f"  Keyword Precision@10: {test_metrics.get('keyword_precision@10', 0.0):.6f}",
                f"  Keyword Recall@10:   {test_metrics.get('keyword_recall@10', 0.0):.6f}",
                f"  Keyword F1@10:        {keyword_f1_10:.6f}",
                f"  Keyword Precision@20: {test_metrics.get('keyword_precision@20', 0.0):.6f}",
                f"  Keyword Recall@20:   {score:.6f} ⭐ (selection metric)",
            ]
            if val_metrics:
                val_keyterm_f1 = 0.0
                if val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0) > 0:
                    val_keyterm_f1 = (2 * val_metrics.get('keyterm_precision@10', 0.0) * val_metrics.get('keyterm_recall@10', 0.0)) / \
                                    (val_metrics.get('keyterm_precision@10', 0.0) + val_metrics.get('keyterm_recall@10', 0.0))
                note_lines += [
                    "",
                    f"Validation Set Performance:",
                    f"  Keyterm F1@10:        {val_keyterm_f1:.6f}",
                ]
            note_lines += [
                "",
                f"Training Configuration:",
                f"  Batch Size: {args.batch_size}",
                f"  Learning Rate: {args.learning_rate}",
                f"  Weight Decay: {args.weight_decay}",
                f"  Gradient Accumulation Steps: {args.grad_accum_steps}",
                f"",
                f"Dataset Configuration:",
                f"  Target Role: {args.target_role}",
                f"  History Turns: {args.history_turns}",
                f"  Max Keywords: {args.max_keywords}",
                f"  Max Keyterms: {args.max_keyterms}",
                f"  Train Examples: {len(train_examples)}",
                f"  Val Examples: {len(val_examples)}",
                f"  Test Examples: {len(test_examples)}",
            ]
            with open(os.path.join(output_dir, "best_note.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(note_lines) + "\n")
            write_json(os.path.join(output_dir, "best_eval.json"), best)
        else:
            print(f"  (No improvement - best Keyword Recall@20 remains {best_score:.6f} from epoch {best['epoch']})")

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
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest Model Summary:")
    print(f"  Epoch:                {best['epoch']}")
    print(f"  Model:                {best.get('model_name', args.model_name)}")
    print(f"  Optimization Goal:   Keyterm Prediction")
    print(f"  Test Keyterm F1@5:    {best.get('keyterm_f1@5', 0.0):.6f}")
    print(f"  Test Keyterm F1@10:   {best.get('keyterm_f1@10', 0.0):.6f}")
    print(f"  Test Keyterm Precision@5:  {best['metrics'].get('keyterm_precision@5', 0.0):.6f}")
    print(f"  Test Keyterm Recall@5:     {best['metrics'].get('keyterm_recall@5', 0.0):.6f}")
    print(f"  Test Keyterm Precision@10: {best['metrics'].get('keyterm_precision@10', 0.0):.6f}")
    print(f"  Test Keyterm Recall@10:    {best['metrics'].get('keyterm_recall@10', 0.0):.6f}")
    print(f"  Test Keyterm Precision@20: {best['metrics'].get('keyterm_precision@20', 0.0):.6f}")
    print(f"  Test Keyterm Recall@20:    {best['metrics'].get('keyterm_recall@20', 0.0):.6f}")
    print(f"  Test Keyword Precision@10: {best['metrics'].get('keyword_precision@10', 0.0):.6f}")
    print(f"  Test Keyword Recall@10:    {best['metrics'].get('keyword_recall@10', 0.0):.6f}")
    print(f"  Test Keyword F1@10:        {best['metrics'].get('keyword_f1@10', 0.0):.6f}")
    print(f"  Test Keyword Precision@20: {best['metrics'].get('keyword_precision@20', 0.0):.6f}")
    print(f"  Test Keyword Recall@20:    {best.get('keyword_recall@20', 0.0):.6f} ⭐ (selection metric)")
    print(f"\nOutput Directory: {output_dir}")
    print(f"Best Model Saved:  {os.path.join(output_dir, 'encoder_best')}")
    print(f"Best Eval JSON:    {os.path.join(output_dir, 'best_eval.json')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
