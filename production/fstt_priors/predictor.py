"""
PriorPredictor — single-class interface for the production team.

Usage:
    from fstt_priors import PriorPredictor

    predictor = PriorPredictor("models/retrieval_minilm_l3_user_only_20260211_001736")
    terms = predictor.predict(conversation_history, max_terms=10, topk=30)
    deepgram_params = predictor.deepgram_params(terms)
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Stopword filter — aggressive list tuned for restaurant phone-order domain.
# Any term composed entirely of these words is dropped before injection.
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset(
    "a about above after again against all am an and any are aren't as at be because been before being below "
    "between both but by can could couldn't did didn't do does doesn't doing don don't down during each few for "
    "from further get got great had hadn't has hasn't have haven't having he her here hers herself him himself "
    "his how i if in into is isn't it its itself just let like ll m me might more most mustn't my myself no nor "
    "not now of off oh ok okay on once only or other our ours ourselves out over own pls please re s same shall "
    "shan't she should shouldn't so some sounds such sure t than thank thanks that that'll the their theirs them "
    "themselves then there these they this those through to too under until up us ve very want was wasn't we well "
    "were weren't what when where which while who whom why will with won won't would wouldn't ya yeah yes yet "
    "you your yours yourself yourselves yay nope think believe need come comes came nice good going go "
    "got know let's look looks make makes much really right see take tell thing things thought try um uh "
    "one two three four five six seven eight nine ten also already actually maybe probably still another "
    "something anything hi hey hello bye sorry wait hang adding order pickup delivery".split()
)

DEEPGRAM_MAX_KEYTERMS = 100


class PriorPredictor:
    """Predict Deepgram biasing terms from conversation history.

    Args:
        model_dir: Path to the model run directory. Must contain:
            - ``best_model/`` — the SentenceTransformer encoder
            - ``shared_index/`` — candidates.json + index files
        device: ``"cpu"`` (default) or ``"cuda"``.

    Example::

        predictor = PriorPredictor("models/retrieval_minilm_l3_user_only_20260211_001736")
        terms = predictor.predict("SYSTEM: Hi\\nUSER: I want a California roll")
        # ["california roll", "salmon", "poke bowl", ...]
    """

    def __init__(self, model_dir: str, device: str = "cpu") -> None:
        self._model_dir = os.path.abspath(model_dir)
        self._device = device

        # Resolve encoder
        encoder_dir = self._find_subdir("best_model")
        self._encoder = SentenceTransformer(encoder_dir, device=device)

        # Resolve index
        index_dir = self._find_subdir("shared_index")
        self._candidates, self._index = self._load_index(index_dir)

    # ── Public API ──────────────────────────────────────────────

    def predict(
        self,
        history: str,
        max_terms: int = 10,
        topk: int = 30,
    ) -> List[str]:
        """Predict biasing terms from a conversation history string.

        Args:
            history: Multi-line conversation transcript, e.g.::

                "SYSTEM: Welcome to Shokudo.\\nUSER: I want a California roll.\\nSYSTEM: Got it."

            max_terms: Maximum terms to return (default 10, capped at 100).
            topk: How many candidates to retrieve from the index (default 30).

        Returns:
            A list of up to ``max_terms`` interleaved keyword+keyterm strings,
            ready to pass as Deepgram ``keyterm`` parameters.
        """
        # Encode
        query_emb = self._encode([history])

        # Retrieve
        retrieved = self._retrieve_texts(query_emb, topk)[0]

        # Parse keywords + keyterms from each retrieved candidate
        all_keywords: List[str] = []
        all_keyterms: List[str] = []
        seen_kw: set = set()
        seen_kt: set = set()

        for cand in retrieved:
            kw, kt = _parse_candidate(cand)
            for w in kw:
                key = w.lower()
                if key not in seen_kw:
                    seen_kw.add(key)
                    all_keywords.append(w)
            for t in kt:
                key = t.lower()
                if key not in seen_kt:
                    seen_kt.add(key)
                    all_keyterms.append(t)

        clean_kw = _filter_stopwords(all_keywords)
        clean_kt = _filter_stopwords(all_keyterms)
        merged = _interleave(clean_kw, clean_kt)

        cap = min(max_terms, DEEPGRAM_MAX_KEYTERMS)
        return merged[:cap]

    @staticmethod
    def deepgram_params(terms: List[str]) -> Dict[str, List[str]]:
        """Build the Deepgram Nova-3 ``keyterm`` query parameter dict.

        Args:
            terms: The list returned by :meth:`predict`.

        Returns:
            ``{"keyterm": [...]}``, ready to pass to the Deepgram SDK::

                from deepgram import DeepgramClient, PrerecordedOptions
                opts = PrerecordedOptions(model="nova-3", **predictor.deepgram_params(terms))
        """
        return {"keyterm": terms}

    @property
    def model_dir(self) -> str:
        """Absolute path to the loaded model directory."""
        return self._model_dir

    # ── Internals ───────────────────────────────────────────────

    def _find_subdir(self, name: str) -> str:
        path = os.path.join(self._model_dir, name)
        if os.path.isdir(path):
            return path
        raise FileNotFoundError(
            f"Expected directory '{name}' inside {self._model_dir}"
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self._encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    def _load_index(self, index_dir: str):
        """Load candidates + build a nearest-neighbor index."""
        cand_path = os.path.join(index_dir, "candidates.json")
        if not os.path.isfile(cand_path):
            raise FileNotFoundError(f"Missing {cand_path}")

        with open(cand_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)

        # Try FAISS first, fall back to sklearn
        faiss_path = os.path.join(index_dir, "index.faiss")
        sklearn_path = os.path.join(index_dir, "index.pkl")

        index = None
        try:
            import faiss
            if os.path.isfile(faiss_path):
                index = faiss.read_index(faiss_path)
        except Exception:
            pass

        if index is None and os.path.isfile(sklearn_path):
            with open(sklearn_path, "rb") as f:
                index = pickle.load(f)
        elif index is None:
            # Rebuild sklearn index from candidates on the fly
            from sklearn.neighbors import NearestNeighbors
            cand_emb = self._encode(candidates)
            nn = NearestNeighbors(metric="cosine", algorithm="brute")
            nn.fit(cand_emb)
            index = nn

        return candidates, index

    def _retrieve_texts(
        self, query_emb: np.ndarray, topk: int
    ) -> List[List[str]]:
        topk = min(topk, len(self._candidates))
        idx = self._index

        # FAISS index
        try:
            import faiss
            if isinstance(idx, faiss.Index):
                _, indices = idx.search(query_emb, topk)
                return [
                    [self._candidates[i] for i in row] for row in indices
                ]
        except Exception:
            pass

        # sklearn NearestNeighbors
        from sklearn.neighbors import NearestNeighbors
        if isinstance(idx, NearestNeighbors):
            _, indices = idx.kneighbors(query_emb, n_neighbors=topk)
            return [
                [self._candidates[i] for i in row] for row in indices
            ]

        raise RuntimeError(f"Unknown index type: {type(idx)}")


# ── Module-level helpers ──────────────────────────────────────


def _parse_candidate(candidate: str) -> Tuple[List[str], List[str]]:
    """Parse 'keyterms: a; b\\nkeywords: x; y' into (keywords, keyterms)."""
    keywords: List[str] = []
    keyterms: List[str] = []
    for line in candidate.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("keyterms:"):
            rhs = line.split(":", 1)[1]
            keyterms = [t.strip() for t in rhs.split(";") if t.strip()]
        elif lower.startswith("keywords:"):
            rhs = line.split(":", 1)[1]
            keywords = [t.strip() for t in rhs.split(";") if t.strip()]
    return keywords, keyterms


def _filter_stopwords(terms: List[str]) -> List[str]:
    """Remove terms composed entirely of stopwords."""
    out: List[str] = []
    for term in terms:
        words = term.lower().split()
        if words and any(w not in _STOPWORDS for w in words):
            out.append(term)
    return out


def _interleave(a: List[str], b: List[str]) -> List[str]:
    """Alternate items from a and b, deduplicating by lowercase."""
    result: List[str] = []
    seen: set = set()
    i, j = 0, 0
    while i < len(a) or j < len(b):
        if i < len(a):
            key = a[i].lower()
            if key not in seen:
                seen.add(key)
                result.append(a[i])
            i += 1
        if j < len(b):
            key = b[j].lower()
            if key not in seen:
                seen.add(key)
                result.append(b[j])
            j += 1
    return result
