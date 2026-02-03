import json
import os
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset

RoleTurn = Tuple[str, str]


def _normalize_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role in {"user", "customer", "human", "client", "guest"}:
        return "USER"
    if role in {"system", "assistant", "agent", "bot", "server"}:
        return "SYSTEM"
    return "SYSTEM"


def _extract_text(turn: Dict) -> str:
    for key in ["text", "utterance", "transcript", "content", "sentence"]:
        if key in turn and isinstance(turn[key], str):
            return turn[key]
    return ""


def _extract_turns_from_item(item: Dict) -> List[RoleTurn]:
    turns: List[RoleTurn] = []

    # DailyDialog (roskoN/dailydialog) style: a single list of utterances.
    # Roles are not explicitly provided; alternate USER/SYSTEM by index.
    if "utterances" in item and isinstance(item["utterances"], list) and item["utterances"]:
        utterances = item["utterances"]
        for i, text in enumerate(utterances):
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            role = "USER" if i % 2 == 0 else "SYSTEM"
            turns.append((role, text))
        if turns:
            return turns

    if "turns" in item and isinstance(item["turns"], dict):
        speakers = item["turns"].get("speaker") or []
        utterances = item["turns"].get("utterance") or []
        for speaker, text in zip(speakers, utterances):
            role = "USER" if str(speaker) == "0" else "SYSTEM"
            text = (text or "").strip()
            if text:
                turns.append((role, text))
        return turns

    turns_list = None
    for key in ["turns", "dialogue", "dialog", "utterances", "messages"]:
        if key in item and isinstance(item[key], list):
            turns_list = item[key]
            break
    if turns_list is None:
        return []

    for idx, turn in enumerate(turns_list):
        if isinstance(turn, str):
            role = "USER" if idx % 2 == 0 else "SYSTEM"
            text = turn
        elif isinstance(turn, dict):
            role = _normalize_role(turn.get("speaker") or turn.get("role") or turn.get("participant") or "")
            text = _extract_text(turn)
            if not text and "utterances" in turn and isinstance(turn["utterances"], str):
                text = turn["utterances"]
        else:
            continue
        text = (text or "").strip()
        if text:
            turns.append((role, text))
    return turns


def load_dialogs(dataset_name: str, split: str, max_dialogs: int) -> List[List[RoleTurn]]:
    if dataset_name not in {"multiwoz", "dailydialog"}:
        raise ValueError("dataset_name must be 'multiwoz' or 'dailydialog'")

    dataset_order = [dataset_name]
    if dataset_name == "multiwoz":
        dataset_order.append("dailydialog")

    last_error = None
    dataset = None
    name_to_repo = {
        "multiwoz": "pfb30/multi_woz_v22",
        # Prefer the HF-hosted DailyDialog mirror (no external zip link).
        "dailydialog": "roskoN/dailydialog",
        # Legacy fallback (may rely on an external zip URL).
        "dailydialog_legacy": "daily_dialog",
    }
    loaded_name = None
    for name in dataset_order:
        try:
            if name == "dailydialog":
                for repo_key in ["dailydialog", "dailydialog_legacy"]:
                    repo_id = name_to_repo[repo_key]
                    try:
                        dataset = load_dataset(repo_id, split=split, trust_remote_code=True)
                        loaded_name = repo_key
                        break
                    except Exception as exc:
                        last_error = exc
                        dataset = None
                if dataset is not None:
                    break
                continue
            repo_id = name_to_repo[name]
            dataset = load_dataset(repo_id, split=split, trust_remote_code=True)
            loaded_name = name
            break
        except Exception as exc:
            last_error = exc
            dataset = None
            continue

    if dataset is None:
        raise RuntimeError(
            f"Failed to load dataset(s) for split '{split}'. "
            f"Last error: {last_error}"
        )
    if loaded_name and loaded_name not in {dataset_name, "dailydialog"}:
        print(f"Warning: falling back to '{loaded_name}' for split '{split}'.")
    if dataset_name == "dailydialog" and loaded_name == "dailydialog_legacy":
        print(f"Warning: using legacy 'daily_dialog' source for split '{split}'. Prefer 'roskoN/dailydialog'.")

    dialogs: List[List[RoleTurn]] = []
    for item in dataset:
        turns = _extract_turns_from_item(item)
        if not turns:
            dialog = item.get("dialog")
            if isinstance(dialog, list):
                turns = [("USER" if i % 2 == 0 else "SYSTEM", t) for i, t in enumerate(dialog) if isinstance(t, str)]
        if turns:
            dialogs.append(turns)
        if max_dialogs and len(dialogs) >= max_dialogs:
            break
    return dialogs


def build_examples(dialogs: Iterable[List[RoleTurn]], history_turns: int) -> List[Dict]:
    examples: List[Dict] = []
    for dialog in dialogs:
        for idx, (role, text) in enumerate(dialog):
            if role != "USER":
                continue
            start = max(0, idx - history_turns)
            history = dialog[start:idx]
            if not history:
                continue
            history_text = "\n".join([f"{r}: {t}" for r, t in history])
            examples.append({
                "history_text": history_text,
                "target_text": text,
            })
    return examples


def write_examples_jsonl(path: str, examples: List[Dict], split: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            record = {"split": split, **ex}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_examples_jsonl(path: str) -> List[Dict]:
    examples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples
