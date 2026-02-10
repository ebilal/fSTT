import argparse
import csv
import os
import sys
import time
from typing import Dict, List

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.eval_shokudo import (  # noqa: E402
    build_eval_examples,
    configure_runtime_threads,
    dedupe_preserve_order,
    discover_run_dirs,
    load_dialogs_csv,
    load_index_with_fallback,
    resolve_encoder_dir,
    resolve_index_dir,
    resolve_input_path,
)
from src.model import encode_texts, load_encoder  # noqa: E402
from src.prior import extract_priors  # noqa: E402
from src.shokudo_eval import (  # noqa: E402
    build_menu_preamble,
    menu_from_json,
    normalize_terms,
    prepare_predictions,
)
from src.utils import ensure_dir, get_device, write_json  # noqa: E402


PROMPT_VARIANTS: List[Dict[str, str]] = [
    {"prompt_id": "plain_history", "prefix": ""},
    {
        "prompt_id": "restaurant_context",
        "prefix": (
            "TASK CONTEXT:\n"
            "You are a restaurant order-taking bot.\n"
            "Predict keywords and keyterms for the next user utterance from prior conversation only.\n"
            "Do not use future utterances."
        ),
    },
    {"prompt_id": "menu_items_only", "prefix": "__MENU_ONLY__"},
    {
        "prompt_id": "restaurant_context_plus_menu",
        "prefix": (
            "TASK CONTEXT:\n"
            "You are a restaurant order-taking bot.\n"
            "Predict keywords and keyterms for the next user utterance from prior conversation only.\n"
            "Do not use future utterances.\n\n"
            "__MENU_ONLY__"
        ),
    },
]


def _prefix_history(history_text: str, prefix_template: str, menu_preamble: str) -> str:
    prefix = prefix_template.replace("__MENU_ONLY__", menu_preamble).strip()
    if not prefix:
        return history_text
    return f"{prefix}\n\n{history_text}"


def write_prompt_sweep_results(out_dir: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(out_dir)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("keyterm_recall_avg", 0.0)),
            float(row.get("keyword_recall_avg", 0.0)),
            -float(row.get("latency_ms_avg", 0.0)),
        ),
        reverse=True,
    )

    csv_path = os.path.join(out_dir, "shokudo_prompt_sweep_summary.csv")
    json_path = os.path.join(out_dir, "shokudo_prompt_sweep_summary.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "prompt_id",
                "run_name",
                "run_dir",
                "resolved_encoder_path",
                "resolved_index_path",
                "keyword_recall_avg",
                "keyterm_recall_avg",
                "latency_ms_avg",
                "latency_ms_p50",
                "latency_ms_p95",
                "num_examples",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)

    write_json(json_path, {"results": [dict(rank=i + 1, **r) for i, r in enumerate(sorted_rows)]})

    print(f"\nPrompt sweep CSV: {csv_path}")
    print(f"Prompt sweep JSON: {json_path}")
    print("\n=== Prompt Sweep Leaderboard ===")
    for rank, row in enumerate(sorted_rows, start=1):
        print(
            f"{rank:2d}. {row['prompt_id']} | {row['run_name']} | "
            f"keyterm={float(row['keyterm_recall_avg']):.4f} | "
            f"keyword={float(row['keyword_recall_avg']):.4f} | "
            f"lat(ms)={float(row['latency_ms_avg']):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast prompt sweep for shokudo retrieval models")
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--menu_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/shokudo_prompt_sweep")
    parser.add_argument("--history_turns", type=int, default=8)
    parser.add_argument("--full_history", type=str, default="false")
    parser.add_argument("--target_role", type=str, default="USER")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--encode_batch_size", type=int, default=128)
    args = parser.parse_args()

    device = get_device(args.device)
    full_history = str(args.full_history).lower() in {"1", "true", "yes", "y"}
    configure_runtime_threads(args.threads)

    dataset_csv = resolve_input_path(args.dataset_csv, "Dataset CSV")
    menu_json = resolve_input_path(args.menu_json, "Menu JSON")

    dialogs = load_dialogs_csv(dataset_csv)
    menu_data = menu_from_json(menu_json)
    menu_preamble = build_menu_preamble(menu_data)
    base_examples = build_eval_examples(
        dialogs=dialogs,
        history_turns=args.history_turns,
        menu_preamble="",
        full_history=full_history,
        target_role=args.target_role,
    )
    if not base_examples:
        raise ValueError("No evaluation examples were generated.")

    gt_keywords_list: List[set] = []
    gt_keyterms_list: List[set] = []
    for ex in base_examples:
        gt = extract_priors([ex["target_text"]], max_keywords=30, max_keyterms=30)
        gt_keywords_list.append(set(normalize_terms(gt["keywords"])))
        gt_keyterms_list.append(set(normalize_terms(gt["keyterms"])))

    run_dirs = dedupe_preserve_order(discover_run_dirs(args.models_dir))
    if not run_dirs:
        raise ValueError(f"No valid runs found under models_dir={args.models_dir}")

    print(f"Using device={device} threads={args.threads} full_history={full_history}")
    print(f"Examples={len(base_examples)} | Prompt variants={len(PROMPT_VARIANTS)} | Runs={len(run_dirs)}")

    all_results: List[Dict[str, object]] = []
    ensure_dir(args.out_dir)

    for run_idx, run_dir in enumerate(run_dirs, start=1):
        print(f"\n{'=' * 80}")
        print(f"Run {run_idx}/{len(run_dirs)}: {run_dir}")
        print(f"{'=' * 80}")

        encoder_path = resolve_encoder_dir(run_dir)
        index_path = resolve_index_dir(run_dir)

        print(f"Loading encoder from {encoder_path} ...", flush=True)
        t0 = time.perf_counter()
        encoder = load_encoder(encoder_path, device=device)
        print(f"Encoder loaded in {time.perf_counter() - t0:.2f}s", flush=True)

        print(f"Loading index from {index_path} ...", flush=True)
        t1 = time.perf_counter()
        index = load_index_with_fallback(index_path, encoder)
        print(f"Index loaded in {time.perf_counter() - t1:.2f}s", flush=True)

        run_name = os.path.basename(os.path.normpath(run_dir))

        for variant in PROMPT_VARIANTS:
            prompt_id = variant["prompt_id"]
            prefix = variant["prefix"]
            print(f"\nPrompt={prompt_id}", flush=True)

            histories = [
                _prefix_history(ex["history_text_plain"], prefix, menu_preamble)
                for ex in base_examples
            ]

            t_start = time.perf_counter()
            hist_emb = encode_texts(encoder, histories, batch_size=args.encode_batch_size)
            encode_ms = (time.perf_counter() - t_start) * 1000.0

            t_retrieve = time.perf_counter()
            retrieved_all = index.retrieve_texts(hist_emb, args.topk)
            retrieve_ms = (time.perf_counter() - t_retrieve) * 1000.0

            latencies_ms: List[float] = []
            keyword_correct = 0
            keyterm_correct = 0
            keyword_total = 0
            keyterm_total = 0

            per_example_latency_ms = (encode_ms + retrieve_ms) / max(1, len(base_examples))
            for ex_idx, retrieved in enumerate(retrieved_all):
                pred = prepare_predictions(retrieved, max_keywords=30, max_keyterms=30)
                pred_keywords = set(normalize_terms(pred["keywords"]))
                pred_keyterms = set(normalize_terms(pred["keyterms"]))
                gt_keywords = gt_keywords_list[ex_idx]
                gt_keyterms = gt_keyterms_list[ex_idx]

                keyword_total += len(gt_keywords)
                keyterm_total += len(gt_keyterms)
                keyword_correct += len(gt_keywords & pred_keywords)
                keyterm_correct += len(gt_keyterms & pred_keyterms)
                latencies_ms.append(per_example_latency_ms)

            lat_array = np.asarray(latencies_ms, dtype="float64")
            result = {
                "prompt_id": prompt_id,
                "run_name": run_name,
                "run_dir": run_dir,
                "resolved_encoder_path": encoder_path,
                "resolved_index_path": index_path,
                "num_examples": len(base_examples),
                "keyword_recall_avg": (keyword_correct / keyword_total) if keyword_total else 0.0,
                "keyterm_recall_avg": (keyterm_correct / keyterm_total) if keyterm_total else 0.0,
                "latency_ms_avg": float(lat_array.mean()) if len(lat_array) else 0.0,
                "latency_ms_p50": float(np.percentile(lat_array, 50)) if len(lat_array) else 0.0,
                "latency_ms_p95": float(np.percentile(lat_array, 95)) if len(lat_array) else 0.0,
            }
            all_results.append(result)
            print(
                f"keyword@30={result['keyword_recall_avg']:.4f} | "
                f"keyterm@30={result['keyterm_recall_avg']:.4f} | "
                f"lat(ms)={result['latency_ms_avg']:.2f}",
                flush=True,
            )

    write_prompt_sweep_results(args.out_dir, all_results)


if __name__ == "__main__":
    main()
