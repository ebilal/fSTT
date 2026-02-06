import json
from typing import Any, Dict, Iterable, List, Tuple


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _flatten_instructions(value: Any) -> List[str]:
    instructions: List[str] = []
    if isinstance(value, str):
        if value.strip():
            instructions.append(value.strip())
        return instructions
    if isinstance(value, list):
        for item in value:
            instructions.extend(_flatten_instructions(item))
        return instructions
    if isinstance(value, dict):
        for item in value.values():
            instructions.extend(_flatten_instructions(item))
    return instructions


def _collect_menu_items(menu_data: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            has_items = isinstance(obj.get("items"), list)
            has_categories = isinstance(obj.get("categories"), list)
            if has_items:
                for entry in obj.get("items", []):
                    visit(entry)
            if has_categories:
                for entry in obj.get("categories", []):
                    visit(entry)
            if has_items or has_categories:
                for key, value in obj.items():
                    if key in {"items", "categories"}:
                        continue
                    visit(value)
                return

            if any(isinstance(obj.get(key), str) for key in ("name", "spoken_name")) or "ordering_instructions" in obj:
                items.append(obj)

            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for entry in obj:
                visit(entry)

    visit(menu_data)
    return items


def build_menu_preamble(menu_data: Any) -> str:
    items = _collect_menu_items(menu_data)
    names: List[str] = []
    asks: List[str] = []

    for item in items:
        for key in ("spoken_name", "name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
        if "ordering_instructions" in item:
            asks.extend(_flatten_instructions(item.get("ordering_instructions")))

    names = _dedupe_preserve(names)
    asks = _dedupe_preserve(asks)

    items_part = ", ".join(names)
    if asks:
        ask_part = "; ".join(asks)
        summary = f"{items_part}; ASK: {ask_part}" if items_part else f"ASK: {ask_part}"
    else:
        summary = items_part

    summary = summary.strip()
    if summary:
        return f"MENU: {summary}"
    return "MENU:"


def flatten_candidate_terms(candidates: Iterable[str]) -> List[str]:
    terms: List[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate)
        if not text.strip():
            continue
        for part in text.split(","):
            term = part.strip()
            if term:
                terms.append(term)
    return _dedupe_preserve(terms)


def split_keywords_keyterms(terms: Iterable[str]) -> Tuple[List[str], List[str]]:
    keywords: List[str] = []
    keyterms: List[str] = []
    for term in terms:
        text = str(term).strip()
        if not text:
            continue
        if len(text.split()) <= 1:
            keywords.append(text)
        else:
            keyterms.append(text)
    return keywords, keyterms


def pad_terms(terms: List[str], size: int) -> List[str]:
    if len(terms) >= size:
        return terms[:size]
    return terms + [""] * (size - len(terms))


def prepare_predictions(candidates: Iterable[str], max_keywords: int = 30, max_keyterms: int = 30) -> Dict[str, List[str]]:
    terms = flatten_candidate_terms(candidates)
    keywords, keyterms = split_keywords_keyterms(terms)
    return {
        "keywords": pad_terms(keywords, max_keywords),
        "keyterms": pad_terms(keyterms, max_keyterms),
    }


def normalize_terms(terms: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for term in terms:
        if term is None:
            continue
        text = str(term).strip().lower()
        if text:
            normalized.append(text)
    return normalized


def compute_recall(gt_terms: Iterable[str], pred_terms: Iterable[str]) -> float:
    gt_set = set(normalize_terms(gt_terms))
    if not gt_set:
        return 0.0
    pred_set = set(normalize_terms(pred_terms))
    return len(gt_set & pred_set) / len(gt_set)


def menu_from_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
