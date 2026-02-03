from typing import Dict, List

import numpy as np

from .index import VectorIndex
from .model import encode_texts


def evaluate_retrieval(
    encoder,
    index: VectorIndex,
    eval_examples: List[dict],
    topk_list: List[int] = None,
) -> Dict[str, float]:
    if topk_list is None:
        topk_list = [1, 5, 10]
    if not eval_examples:
        return {f"recall@{k}": 0.0 for k in topk_list} | {"mrr@10": 0.0}

    max_k = max(topk_list)
    histories = [ex["history_text"] for ex in eval_examples]
    targets = [ex["target_text"] for ex in eval_examples]

    query_embeddings = encode_texts(encoder, histories)
    indices, _ = index.retrieve(query_embeddings, max_k)

    recalls = {k: 0 for k in topk_list}
    mrr_total = 0.0
    for i, row in enumerate(indices):
        retrieved_texts = [index.target_texts[idx] for idx in row]
        target = targets[i]
        for k in topk_list:
            if target in retrieved_texts[:k]:
                recalls[k] += 1
        rr = 0.0
        for rank, text in enumerate(retrieved_texts[:10], start=1):
            if text == target:
                rr = 1.0 / rank
                break
        mrr_total += rr

    total = len(eval_examples)
    metrics = {f"recall@{k}": recalls[k] / total for k in topk_list}
    metrics["mrr@10"] = mrr_total / total
    return metrics


def sample_qualitative(
    encoder,
    index: VectorIndex,
    eval_examples: List[dict],
    topk: int = 5,
    num_samples: int = 5,
) -> List[Dict]:
    samples = eval_examples[:num_samples]
    histories = [ex["history_text"] for ex in samples]
    query_embeddings = encode_texts(encoder, histories)
    indices, _ = index.retrieve(query_embeddings, topk)
    results = []
    for ex, row in zip(samples, indices):
        results.append({
            "history_text": ex["history_text"],
            "target_text": ex["target_text"],
            "retrieved": [index.target_texts[idx] for idx in row],
        })
    return results

