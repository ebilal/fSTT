import pytest

from src.data import build_examples, load_dialogs


def test_build_examples_smoke():
    try:
        dialogs = load_dialogs("dailydialog", split="train", max_dialogs=2)
    except Exception as exc:
        pytest.skip(f"Dataset unavailable: {exc}")
    examples = build_examples(dialogs, history_turns=2)
    assert isinstance(examples, list)
    assert all("history_text" in ex and "target_text" in ex for ex in examples)

