"""Example integration for LiveKit Agents + Deepgram priors.

This module avoids hard dependencies on LiveKit; the snippet can be copied into a
LiveKit agent project with the proper dependencies installed.
"""

from typing import Dict, List

from .prior import build_deepgram_extra_kwargs


def get_livekit_snippet() -> str:
    return """
# Minimal LiveKit Agents snippet showing Deepgram extra_kwargs injection
from livekit.agents import VoicePipelineAgent
from livekit.agents.stt import DeepgramSTT

from src.prior import build_deepgram_extra_kwargs
from src.utils import resolve_run_dir
from src.model import load_encoder, encode_texts
from src.index import VectorIndex
from src.prior import extract_priors

# Load artifacts once at agent startup
run_dir = resolve_run_dir("outputs/latest")
encoder = load_encoder(f"{run_dir}/encoder", device="cpu")
index = VectorIndex.load(f"{run_dir}/index")

history_buffer = []  # rolling list of (role, text)

def update_history(role: str, text: str, max_turns: int = 6) -> None:
    history_buffer.append((role, text))
    if len(history_buffer) > max_turns:
        history_buffer[:] = history_buffer[-max_turns:]


def history_text() -> str:
    return "\n".join([f"{r}: {t}" for r, t in history_buffer])


def predict_prior() -> dict:
    if not history_buffer:
        return {"keywords": [], "keyterms": []}
    embedding = encode_texts(encoder, [history_text()])
    retrieved = index.retrieve_texts(embedding, topk=5)[0]
    return extract_priors(retrieved)

# During a new user turn
prior = predict_prior()
extra_kwargs = build_deepgram_extra_kwargs(prior, deepgram_model="nova-3")

stt = DeepgramSTT(model="nova-3", api_key="${DEEPGRAM_API_KEY}", extra_kwargs=extra_kwargs)
agent = VoicePipelineAgent(stt=stt)
""".strip()


def build_extra_kwargs_example(prior: Dict[str, List[str]], deepgram_model: str) -> Dict[str, List[str]]:
    return build_deepgram_extra_kwargs(prior, deepgram_model)


if __name__ == "__main__":
    print(get_livekit_snippet())

