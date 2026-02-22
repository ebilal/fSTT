import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import faiss

    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False


def faiss_available() -> bool:
    return _FAISS_AVAILABLE


def _write_jsonl(path: str, items: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({"text": item}, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line)["text"])
    return items


class VectorIndex:
    def __init__(self, index_type: str, target_texts: List[str], dim: int):
        self.index_type = index_type
        self.target_texts = target_texts
        self.dim = dim
        self.faiss_index = None
        self.nn_index = None

    @classmethod
    def build(cls, embeddings: np.ndarray, target_texts: List[str], prefer_faiss: bool = True) -> "VectorIndex":
        dim = embeddings.shape[1]
        use_faiss = prefer_faiss and faiss_available()
        if use_faiss:
            index = cls(index_type="faiss", target_texts=target_texts, dim=dim)
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(embeddings.astype("float32"))
            index.faiss_index = faiss_index
            return index
        index = cls(index_type="sklearn", target_texts=target_texts, dim=dim)
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(embeddings)
        index.nn_index = nn
        return index

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        meta_path = os.path.join(dir_path, "index_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "index_type": self.index_type,
                "dim": self.dim,
                "num_targets": len(self.target_texts),
            }, f, indent=2)
        _write_jsonl(os.path.join(dir_path, "targets.jsonl"), self.target_texts)
        if self.index_type == "faiss":
            faiss.write_index(self.faiss_index, os.path.join(dir_path, "index.faiss"))
        else:
            with open(os.path.join(dir_path, "index.pkl"), "wb") as f:
                pickle.dump(self.nn_index, f)

    @classmethod
    def load(cls, dir_path: str) -> "VectorIndex":
        meta_path = os.path.join(dir_path, "index_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        target_texts = _read_jsonl(os.path.join(dir_path, "targets.jsonl"))
        index = cls(meta["index_type"], target_texts, meta["dim"])
        if index.index_type == "faiss":
            index.faiss_index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        else:
            with open(os.path.join(dir_path, "index.pkl"), "rb") as f:
                index.nn_index = pickle.load(f)
        return index

    @classmethod
    def load_shared_index(cls, dir_path: str) -> "VectorIndex":
        """Load from shared_index format: candidates.json + index.faiss (+ meta.json)."""
        if not faiss_available():
            raise RuntimeError("faiss required for load_shared_index")
        cand_path = os.path.join(dir_path, "candidates.json")
        faiss_path = os.path.join(dir_path, "index.faiss")
        if not os.path.isfile(cand_path):
            raise FileNotFoundError(f"Missing {cand_path}")
        if not os.path.isfile(faiss_path):
            raise FileNotFoundError(f"Missing {faiss_path}")
        with open(cand_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        target_texts = (
            loaded.get("candidates", loaded)
            if isinstance(loaded, dict)
            else loaded
        )
        if not isinstance(target_texts, list):
            raise ValueError(f"Unexpected candidates.json structure in {dir_path}")
        target_texts = [str(t) for t in target_texts]
        faiss_index = faiss.read_index(faiss_path)
        dim = int(faiss_index.d)
        index = cls(index_type="faiss", target_texts=target_texts, dim=dim)
        index.faiss_index = faiss_index
        return index

    def retrieve(self, query_embeddings: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.target_texts) == 0:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)
        topk = min(topk, len(self.target_texts))
        if self.index_type == "faiss":
            scores, indices = self.faiss_index.search(query_embeddings.astype("float32"), topk)
            return indices, scores
        distances, indices = self.nn_index.kneighbors(query_embeddings, n_neighbors=topk, return_distance=True)
        scores = 1.0 - distances
        return indices, scores

    def retrieve_texts(self, query_embeddings: np.ndarray, topk: int) -> List[List[str]]:
        indices, _ = self.retrieve(query_embeddings, topk)
        results: List[List[str]] = []
        for row in indices:
            results.append([self.target_texts[i] for i in row])
        return results

