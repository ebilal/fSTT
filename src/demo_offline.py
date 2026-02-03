import argparse
import json
import os

from .data import load_examples_jsonl
from .index import VectorIndex
from .model import load_encoder, encode_texts
from .prior import extract_priors
from .utils import get_device, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval + prior demo")
    parser.add_argument("--run", type=str, default="outputs/latest")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    device = get_device(args.device)

    encoder = load_encoder(os.path.join(run_dir, "encoder"), device=device)
    index = VectorIndex.load(os.path.join(run_dir, "index"))

    examples_path = os.path.join(run_dir, "examples.jsonl")
    examples = load_examples_jsonl(examples_path)
    if not examples:
        raise ValueError("No examples found in examples.jsonl")

    example = examples[0]
    history = example["history_text"]

    query_embedding = encode_texts(encoder, [history])
    retrieved = index.retrieve_texts(query_embedding, args.topk)[0]
    prior = extract_priors(retrieved)

    print("=== History ===")
    print(history)
    print("\n=== Retrieved Candidates ===")
    for idx, text in enumerate(retrieved, start=1):
        print(f"{idx}. {text}")
    print("\n=== Extracted Priors ===")
    print(json.dumps(prior, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

