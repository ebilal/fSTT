import re
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_NUMERIC_RE = re.compile(r"\b\d{1,4}(?::\d{2})?\b")
_TIME_RE = re.compile(r"\b\d{1,2}(?:am|pm)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\b", re.IGNORECASE)


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _tfidf_top_terms(texts: List[str], ngram_range: tuple, max_items: int) -> List[str]:
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range, min_df=1)
    tfidf = vectorizer.fit_transform(texts)
    if tfidf.shape[1] == 0:
        return []
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    ranked = list(zip(terms, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:max_items]]


def extract_priors(candidates: List[str], max_keywords: int = 30, max_keyterms: int = 30) -> Dict[str, List[str]]:
    if not candidates:
        return {"keywords": [], "keyterms": []}

    keywords = _tfidf_top_terms(candidates, (1, 1), max_keywords)
    keyterms = _tfidf_top_terms(candidates, (2, 3), max_keyterms)

    numeric_terms = []
    for text in candidates:
        numeric_terms.extend(_NUMERIC_RE.findall(text))
        numeric_terms.extend(_TIME_RE.findall(text))
        numeric_terms.extend(_DATE_RE.findall(text))

    keywords = _dedupe(keywords + numeric_terms)
    keyterms = _dedupe(keyterms)

    return {
        "keywords": keywords[:max_keywords],
        "keyterms": keyterms[:max_keyterms],
    }


def build_deepgram_extra_kwargs(prior: Dict[str, List[str]], deepgram_model: str) -> Dict[str, List[str]]:
    model = (deepgram_model or "").lower()
    if "nova-3" in model:
        return {"keyterm": prior.get("keyterms", [])[:100]}
    return {"keywords": prior.get("keywords", [])[:100]}
