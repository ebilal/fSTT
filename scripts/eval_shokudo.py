import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Threading defaults are configured at runtime based on CLI args.

import numpy as np

from src.index import VectorIndex, faiss_available
from src.model import encode_texts, load_encoder
from src.prior import extract_priors
from src.shokudo_eval import (
    build_menu_preamble,
    menu_from_json,
    normalize_terms,
    prepare_predictions,
)
from src.utils import ensure_dir, get_device, resolve_run_dir, write_json

try:
    import faiss
except Exception:
    faiss = None


def _safe_sort_key(value: str, tie_breaker: int) -> Tuple[int, object, int]:
    try:
        return (0, int(value), tie_breaker)
    except Exception:
        return (1, str(value), tie_breaker)


def load_dialogs_csv(path: str) -> Dict[str, List[Dict[str, str]]]:
    dialogs: Dict[str, List[Dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dialog_id = (row.get("dialog_id") or "").strip()
            if not dialog_id:
                continue
            entry = {
                "utterance_id": (row.get("utterance_id") or "").strip(),
                "speaker": (row.get("speaker") or "").strip(),
                "text": (row.get("text") or "").strip(),
                "_row": idx,
            }
            if not entry["text"]:
                continue
            dialogs.setdefault(dialog_id, []).append(entry)

    for dialog_id, turns in dialogs.items():
        turns.sort(key=lambda t: _safe_sort_key(t["utterance_id"], t["_row"]))
    return dialogs


def normalize_speaker(speaker: str) -> str:
    speaker_norm = (speaker or "").strip().lower()
    if speaker_norm in {"agent", "assistant", "system", "bot", "server"}:
        return "SYSTEM"
    if speaker_norm in {"customer", "user", "human", "client", "guest"}:
        return "USER"
    return (speaker or "").strip().upper() or "SYSTEM"


def build_eval_examples(
    dialogs: Dict[str, List[Dict[str, str]]],
    history_turns: int,
    menu_preamble: str,
    full_history: bool = True,
    target_role: str = "USER",
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    target_role = (target_role or "USER").strip().upper()
    for dialog_id in sorted(dialogs.keys(), key=str):
        turns = dialogs[dialog_id]
        if len(turns) < 2:
            continue
        for idx in range(len(turns) - 1):
            if full_history:
                start = 0
            else:
                start = max(0, idx - history_turns + 1)
            history = turns[start:idx + 1]
            if not history:
                continue
            history_lines = [
                f"{normalize_speaker(t['speaker'])}: {t['text']}" for t in history
            ]
            history_text_multiline = "\n".join(history_lines)
            history_text_plain = history_text_multiline
            history_text = history_text_multiline
            if menu_preamble:
                history_text = menu_preamble + "\n" + history_text_multiline
            target_turn = turns[idx + 1]
            if normalize_speaker(target_turn["speaker"]) != target_role:
                continue
            examples.append({
                "dialog_id": dialog_id,
                "utterance_id": target_turn["utterance_id"],
                "history_text": history_text,
                "history_text_plain": history_text_plain,
                "target_text": target_turn["text"],
            })
    return examples


def _looks_like_encoder_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = {"modules.json", "sentence_bert_config.json", "config.json"}
    return any(os.path.exists(os.path.join(path, name)) for name in markers)


def resolve_encoder_dir(run_dir: str) -> str:
    candidates = [
        "best_model",
        "encoder_best",
        "encoder",
        "model",
        "sentence_transformer",
        "sentence-transformer",
        "encoder_model",
    ]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if _looks_like_encoder_dir(path):
            return path
    for name in candidates:
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            return path

    for root, dirs, files in os.walk(run_dir):
        depth = os.path.relpath(root, run_dir).count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        if {"modules.json", "sentence_bert_config.json"}.intersection(files):
            return root
    raise FileNotFoundError(f"No encoder directory found under {run_dir}")


def _is_legacy_index_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "index_meta.json"))
        and os.path.exists(os.path.join(path, "targets.jsonl"))
    )


def _is_shared_index_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.exists(os.path.join(path, "meta.json"))
        and os.path.exists(os.path.join(path, "candidates.json"))
        and os.path.exists(os.path.join(path, "index.faiss"))
    )


def resolve_index_dir(run_dir: str) -> str:
    candidates = [
        "shared_index",
        "index",
        "index_best",
        "index_eval",
        "index_final",
    ]
    for name in candidates:
        path = os.path.join(run_dir, name)
        if _is_legacy_index_dir(path) or _is_shared_index_dir(path):
            return path
    for root, dirs, files in os.walk(run_dir):
        depth = os.path.relpath(root, run_dir).count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        if "index_meta.json" in files:
            return root
        if {"meta.json", "candidates.json", "index.faiss"}.issubset(set(files)):
            return root
    raise FileNotFoundError(f"No index directory found under {run_dir}")


def _read_targets_jsonl(path: str) -> List[str]:
    targets: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                targets.append(text.strip())
    return targets


def load_index_with_fallback(index_dir: str, encoder) -> VectorIndex:
    if os.path.exists(os.path.join(index_dir, "index_meta.json")):
        try:
            return VectorIndex.load(index_dir)
        except AttributeError:
            if faiss_available():
                raise
            meta_path = os.path.join(index_dir, "index_meta.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("index_type") != "faiss":
                raise

            print(f"Warning: FAISS index found but faiss is unavailable; rebuilding sklearn index from targets in {index_dir}")
            targets = _read_targets_jsonl(os.path.join(index_dir, "targets.jsonl"))
            target_emb = encode_texts(encoder, targets, batch_size=256)
            return VectorIndex.build(target_emb, targets, prefer_faiss=False)

    if os.path.exists(os.path.join(index_dir, "meta.json")) and os.path.exists(os.path.join(index_dir, "candidates.json")):
        with open(os.path.join(index_dir, "candidates.json"), "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            target_texts = loaded.get("candidates", [])
        else:
            target_texts = loaded
        if not isinstance(target_texts, list):
            raise ValueError(f"Unexpected candidates.json structure in {index_dir}")
        target_texts = [str(item) for item in target_texts]

        faiss_path = os.path.join(index_dir, "index.faiss")
        if os.path.exists(faiss_path):
            if faiss is None:
                print(
                    f"Warning: shared_index has FAISS index but faiss is unavailable; rebuilding sklearn index from candidates in {index_dir}"
                )
                target_emb = encode_texts(encoder, target_texts, batch_size=256)
                return VectorIndex.build(target_emb, target_texts, prefer_faiss=False)
            dim = encoder.get_sentence_embedding_dimension()
            index = VectorIndex(index_type="faiss", target_texts=target_texts, dim=dim)
            index.faiss_index = faiss.read_index(faiss_path)
            return index

        target_emb = encode_texts(encoder, target_texts, batch_size=256)
        return VectorIndex.build(target_emb, target_texts, prefer_faiss=False)

    raise FileNotFoundError(f"No supported index format found in {index_dir}")


def discover_run_dirs(models_dir: str) -> List[str]:
    if not os.path.isdir(models_dir):
        return []
    discovered: List[str] = []
    for name in sorted(os.listdir(models_dir)):
        run_path = os.path.join(models_dir, name)
        if not os.path.isdir(run_path):
            continue
        try:
            resolve_encoder_dir(run_path)
            resolve_index_dir(run_path)
            discovered.append(run_path)
        except FileNotFoundError:
            continue
    return discovered


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def write_aggregate_results(out_dir: str, results: List[Dict[str, object]]) -> None:
    ensure_dir(out_dir)
    sorted_rows = sorted(
        results,
        key=lambda row: (
            float(row.get("keyterm_recall_avg", 0.0)),
            float(row.get("keyword_recall_avg", 0.0)),
        ),
        reverse=True,
    )
    all_csv_path = os.path.join(out_dir, "shokudo_eval_summary_all_runs.csv")
    all_json_path = os.path.join(out_dir, "shokudo_eval_summary_all_runs.json")

    with open(all_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "run_dir",
                "resolved_encoder_path",
                "resolved_index_path",
                "num_examples",
                "keyword_recall_avg",
                "keyterm_recall_avg",
                "latency_ms_avg",
                "latency_ms_p50",
                "latency_ms_p95",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted_rows)
    write_json(all_json_path, {"results": sorted_rows})

    print(f"All-run summary CSV: {all_csv_path}")
    print(f"All-run summary JSON: {all_json_path}")


def print_leaderboard(results: List[Dict[str, object]]) -> None:
    sorted_rows = sorted(
        results,
        key=lambda row: (
            float(row.get("keyterm_recall_avg", 0.0)),
            float(row.get("keyword_recall_avg", 0.0)),
        ),
        reverse=True,
    )
    print("\n=== Leaderboard (sorted by keyterm recall, then keyword recall) ===")
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:2d}. {row.get('run_name', '<unknown>')} | "
            f"keyterm={float(row.get('keyterm_recall_avg', 0.0)):.4f} | "
            f"keyword={float(row.get('keyword_recall_avg', 0.0)):.4f} | "
            f"lat(ms)={float(row.get('latency_ms_avg', 0.0)):.2f}"
        )


def configure_runtime_threads(threads: int) -> None:
    threads = max(1, int(threads))

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass

    if faiss_available():
        try:
            import faiss

            faiss.omp_set_num_threads(threads)
        except Exception:
            pass


def resolve_input_path(path: str, label: str) -> str:
    if os.path.exists(path):
        return path
    basename = os.path.basename(path)
    candidates = [
        os.path.join(ROOT_DIR, "examples", basename),
        os.path.join(ROOT_DIR, basename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"Warning: {label} not found at {path}; using {candidate} instead.")
            return candidate
    raise FileNotFoundError(f"{label} not found at {path} and no fallback candidate exists.")


def evaluate_run(
    run_dir: str,
    examples: List[Dict[str, str]],
    out_dir: str,
    device: str,
    topk: int,
    index_source: str,
) -> Dict[str, object]:
    resolved_run = resolve_run_dir(run_dir)
    encoder_path = resolve_encoder_dir(resolved_run)
    index_path = resolve_index_dir(resolved_run)

    print(f"Loading encoder from {encoder_path}...", flush=True)
    load_start = time.perf_counter()
    encoder = load_encoder(encoder_path, device=device)
    encoder_load_time = time.perf_counter() - load_start
    print(f"Encoder loaded in {encoder_load_time:.2f}s. Loading index from {index_path}...", flush=True)
    index_start = time.perf_counter()
    if index_source == "run":
        index = load_index_with_fallback(index_path, encoder)
    elif index_source == "shokudo":
        target_texts = [ex["target_text"] for ex in examples]
        target_emb = encode_texts(encoder, target_texts)
        index = VectorIndex.build(target_emb, target_texts, prefer_faiss=True)
        index_path = "<shokudo_eval_index>"
    else:
        raise ValueError(f"Unknown index_source: {index_source}")
    index_load_time = time.perf_counter() - index_start
    print(f"Index loaded in {index_load_time:.2f}s. Processing {len(examples)} examples...", flush=True)

    latencies: List[float] = []
    rows: List[Dict[str, str]] = []
    keyword_correct = 0
    keyword_total = 0
    keyterm_correct = 0
    keyterm_total = 0

    total_examples = len(examples)
    progress_interval = max(1, total_examples // 20)  # Print progress ~20 times
    eval_start = time.perf_counter()
    
    for idx, ex in enumerate(examples):
        if idx % progress_interval == 0 or idx == total_examples - 1:
            elapsed = time.perf_counter() - eval_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_examples - idx - 1) / rate if rate > 0 else 0
            print(f"Processing example {idx + 1}/{total_examples} ({100.0 * (idx + 1) / total_examples:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} ex/s | ETA: {eta:.1f}s", flush=True)
        history_text = ex["history_text"]
        history_text_plain = ex.get("history_text_plain", "")

        start = time.perf_counter()
        embedding = encode_texts(encoder, [history_text])
        retrieved = index.retrieve_texts(embedding, topk)[0]
        pred = prepare_predictions(retrieved, max_keywords=30, max_keyterms=30)
        latency_ms = (time.perf_counter() - start) * 1000.0

        latencies.append(latency_ms)

        gt = extract_priors([ex["target_text"]], max_keywords=30, max_keyterms=30)

        gt_keywords = set(normalize_terms(gt["keywords"]))
        gt_keyterms = set(normalize_terms(gt["keyterms"]))
        pred_keywords = set(normalize_terms(pred["keywords"]))
        pred_keyterms = set(normalize_terms(pred["keyterms"]))

        keyword_total += len(gt_keywords)
        keyterm_total += len(gt_keyterms)
        keyword_correct += len(gt_keywords & pred_keywords)
        keyterm_correct += len(gt_keyterms & pred_keyterms)

        rows.append({
            "dialog_id": ex["dialog_id"],
            "utterance_id": ex["utterance_id"],
            "history_text": history_text_plain,
            "next_utterance": ex["target_text"],
            "ground_truth_keywords": json.dumps(gt["keywords"], ensure_ascii=False),
            "pred_keywords_30": json.dumps(pred["keywords"], ensure_ascii=False),
            "ground_truth_keyterms": json.dumps(gt["keyterms"], ensure_ascii=False),
            "pred_keyterms_30": json.dumps(pred["keyterms"], ensure_ascii=False),
            "latency_ms": f"{latency_ms:.3f}",
        })

    eval_time = time.perf_counter() - eval_start
    print(f"\nCompleted processing {total_examples} examples in {eval_time:.2f}s.", flush=True)
    print(f"Writing results to {out_dir}...", flush=True)

    keyword_recall_avg = keyword_correct / keyword_total if keyword_total else 0.0
    keyterm_recall_avg = keyterm_correct / keyterm_total if keyterm_total else 0.0

    lat_array = np.asarray(latencies, dtype="float64") if latencies else np.array([])
    latency_ms_avg = float(lat_array.mean()) if latencies else 0.0
    latency_ms_p50 = float(np.percentile(lat_array, 50)) if latencies else 0.0
    latency_ms_p95 = float(np.percentile(lat_array, 95)) if latencies else 0.0

    run_name = os.path.basename(os.path.normpath(resolved_run)) or "run"
    ensure_dir(out_dir)
    preds_path = os.path.join(out_dir, f"shokudo_eval_predictions_{run_name}.csv")
    summary_path = os.path.join(out_dir, f"shokudo_eval_summary_{run_name}.json")

    with open(preds_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dialog_id",
                "utterance_id",
                "history_text",
                "next_utterance",
                "ground_truth_keywords",
                "pred_keywords_30",
                "ground_truth_keyterms",
                "pred_keyterms_30",
                "latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, object] = {
        "run_name": run_name,
        "run_dir": resolved_run,
        "resolved_encoder_path": encoder_path,
        "resolved_index_path": index_path,
        "num_examples": len(examples),
        "keyword_recall_avg": keyword_recall_avg,
        "keyterm_recall_avg": keyterm_recall_avg,
        "latency_ms_avg": latency_ms_avg,
        "latency_ms_p50": latency_ms_p50,
        "latency_ms_p95": latency_ms_p95,
    }
    write_json(summary_path, summary)

    print("=== Shokudo Eval ===")
    print(f"Run: {resolved_run}")
    print(f"Encoder: {encoder_path}")
    print(f"Index: {index_path}")
    print(f"Examples: {len(examples)}")
    print(f"Keyword Recall@30 (micro): {keyword_recall_avg:.4f}")
    print(f"Keyterm Recall@30 (micro): {keyterm_recall_avg:.4f}")
    print(f"Latency ms avg/p50/p95: {latency_ms_avg:.2f} / {latency_ms_p50:.2f} / {latency_ms_p95:.2f}")
    print(f"Predictions CSV: {preds_path}")
    print(f"Summary JSON: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Shokudo next-utterance keyterms/keywords predictor")
    parser.add_argument("--run_dir", action="append", help="Run directory (repeatable)")
    parser.add_argument("--models_dir", type=str, default=None, help="Auto-discover run directories under this root")
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--menu_json", type=str, required=True)
    parser.add_argument(
        "--include_menu",
        type=str,
        default="false",
        help="Include menu preamble in input (true/false). Default false.",
    )
    parser.add_argument("--history_turns", type=int, default=6)
    parser.add_argument("--full_history", type=str, default="true", help="Use full conversation history up to current turn")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for FAISS/BLAS/PyTorch")
    parser.add_argument("--target_role", type=str, default="USER", help="Only evaluate when next utterance is this role")
    parser.add_argument(
        "--index_source",
        type=str,
        default="run",
        choices=["run", "shokudo"],
        help="Index source: 'run' loads the saved index, 'shokudo' builds a closed-set index from eval targets.",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}", flush=True)
    configure_runtime_threads(args.threads)
    print(f"Using threads: {max(1, int(args.threads))}", flush=True)
    full_history = args.full_history.lower() == "true"
    print(f"Using full history: {full_history}", flush=True)

    dataset_csv = resolve_input_path(args.dataset_csv, "Dataset CSV")
    menu_json = resolve_input_path(args.menu_json, "Menu JSON")
    include_menu = str(args.include_menu).lower() in {"1", "true", "yes", "y"}
    menu_preamble = ""
    if include_menu:
        print(f"Loading menu from {menu_json}...", flush=True)
        menu_data = menu_from_json(menu_json)
        menu_preamble = build_menu_preamble(menu_data)

    print(f"Loading dialogs from {dataset_csv}...", flush=True)
    dialogs = load_dialogs_csv(dataset_csv)
    print(f"Loaded {len(dialogs)} dialogs. Building evaluation examples...", flush=True)
    examples = build_eval_examples(
        dialogs,
        args.history_turns,
        menu_preamble,
        full_history=full_history,
        target_role=args.target_role,
    )
    print(f"Built {len(examples)} evaluation examples.", flush=True)

    if not examples:
        raise ValueError("No evaluation examples produced from dataset.")

    discovered_runs = discover_run_dirs(args.models_dir) if args.models_dir else []
    explicit_runs = list(args.run_dir or [])
    run_dirs = dedupe_preserve_order(explicit_runs + discovered_runs)
    if not run_dirs:
        raise ValueError("No runs to evaluate. Provide --run_dir and/or --models_dir.")

    print(f"Evaluating {len(run_dirs)} run(s).", flush=True)
    if args.models_dir:
        print(f"Discovered {len(discovered_runs)} run(s) from models dir: {args.models_dir}", flush=True)

    all_results: List[Dict[str, object]] = []
    for run_idx, run_dir in enumerate(run_dirs):
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating run {run_idx + 1}/{len(run_dirs)}: {run_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)
        result = evaluate_run(
            run_dir=run_dir,
            examples=examples,
            out_dir=args.out_dir,
            device=device,
            topk=args.topk,
            index_source=args.index_source,
        )
        all_results.append(result)

    write_aggregate_results(args.out_dir, all_results)
    print_leaderboard(all_results)


if __name__ == "__main__":
    main()
