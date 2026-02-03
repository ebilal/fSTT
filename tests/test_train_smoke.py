import os

import pytest

from src.data import build_examples, load_dialogs
from src.index import VectorIndex
from src.model import train_biencoder, encode_texts


def test_train_pipeline_smoke(tmp_path):
    try:
        dialogs = load_dialogs("dailydialog", split="train", max_dialogs=2)
    except Exception as exc:
        pytest.skip(f"Dataset unavailable: {exc}")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as exc:
        pytest.skip(f"SentenceTransformers unavailable: {exc}")

    examples = build_examples(dialogs, history_turns=2)
    if len(examples) < 2:
        pytest.skip("Not enough examples for smoke training")

    encoder_dir = tmp_path / "encoder"
    model = train_biencoder(
        train_examples=examples,
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        device="cpu",
        epochs=1,
        batch_size=2,
        output_dir=str(encoder_dir),
    )

    targets = [ex["target_text"] for ex in examples]
    embeddings = encode_texts(model, targets, batch_size=4)
    index = VectorIndex.build(embeddings, targets, prefer_faiss=False)
    index_dir = tmp_path / "index"
    index.save(str(index_dir))

    assert os.path.exists(index_dir / "index_meta.json")

