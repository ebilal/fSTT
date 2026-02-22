"""
Deepgram STT keyword/keyterm injection using the retrieval model.

Recommended configuration (based on evaluation on 8 real Shokudo conversations):
  - Strategy:  keywords_plus_keyterms (interleaved)
  - Cap:       10 terms
  - Retrieval: top-30 candidates from the shared index
  - Model:     Deepgram Nova-3 using the `keyterm` parameter

This produced 2 clear transcription wins (correcting menu item names like
"tonkotsu ramen" and "poke bowl") with 0 regressions across all test calls.

Usage (standalone):
    python -m src.livekit_deepgram_inject \\
        --run_dir models/retrieval_minilm_l3_user_only_20260211_001736 \\
        --history "SYSTEM: What can I get you?\\nUSER: I'd like a ramen."

Usage (in a LiveKit agent — see get_livekit_snippet()).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Stopword filter — matches the training-time list so predicted terms that
# slipped through are caught at inference too.
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

# Deepgram documents a maximum of 100 keyterms per request.
DEEPGRAM_MAX_KEYTERMS = 100

# Recommended default: 10 interleaved terms gave the best win/regression ratio.
DEFAULT_MAX_TERMS = 10
DEFAULT_TOPK = 30


def filter_stopwords(terms: List[str]) -> List[str]:
    """Remove terms that are empty or composed entirely of stopwords."""
    out: List[str] = []
    for term in terms:
        words = term.lower().split()
        if not words:
            continue
        if any(w not in _STOPWORDS for w in words):
            out.append(term)
    return out


# Substrings that indicate venue/business names (biases Deepgram to mishear).
_MISLEADING_SUBSTRINGS = (
    "restaurant",
    "noodle bar",
    " bar",
)

# Person names and other-domain terms that degrade food-order transcription.
_BLOCKED_WORDS = frozenset({
    # Common first/last names (injecting biases Deepgram to mishear real names)
    "jim", "mary", "steve", "john", "mike", "david", "sarah", "bob", "johnson",
    "smith", "chen", "wong", "yongmei", "allen", "bell", "tom", "jane", "sir",
    "james", "nicholas", "martin", "benjamin", "george", "iris", "walter",
    "madison", "jenny", "clinton", "daniel", "patrick", "kevin", "jason",
    "paula", "paul", "lucy", "charles", "lee", "peter", "claire", "linda",
    "donald", "alex", "sam", "chris", "mark", "ryan", "nick", "lisa",
    "mac", "barbara", "bradley", "henry", "joann", "obama", "kelly",
    # Other domains (train, taxi, hotel, cinema, hospital)
    "taxi", "train", "trains", "departing", "leaving", "hotel", "cinema",
    "cineworld", "hospital", "appointment", "postcode", "reference", "booking",
    # Address fragments / places
    "riverside", "wilton", "cambridge", "tokyo",
})

# Pattern: mostly digits (phone numbers, IDs)
_RE_MOSTLY_DIGITS = re.compile(r"^[\d\s\-:\.]+$")
# Pattern: alphanumeric code (postcodes, ref IDs: cb21ad, v5weda1v)
_RE_CODE_LIKE = re.compile(r"^[a-z0-9]{5,12}$", re.I)


def _looks_like_code(term: str) -> bool:
    """True if term looks like a postcode, ref ID, or similar."""
    t = term.strip()
    if len(t) < 5:
        return False
    # All digits with optional separators = phone/time
    if _RE_MOSTLY_DIGITS.match(t.replace(" ", "").replace("-", "").replace(":", "")):
        return True
    # Short alphanumeric = postcode or ref
    if _RE_CODE_LIKE.match(t) and sum(c.isdigit() for c in t) >= 1:
        return True
    return False


def filter_misleading_terms(terms: List[str]) -> List[str]:
    """Exclude terms that could bias Deepgram toward wrong words."""
    out: List[str] = []
    for term in terms:
        lower = term.lower().strip()
        if not lower:
            continue
        # Venue-name substrings
        if any(sub in lower for sub in _MISLEADING_SUBSTRINGS):
            continue
        # Blocked words (names, other-domain)
        if lower in _BLOCKED_WORDS:
            continue
        # Multi-word: block if every word is blocked
        words = lower.split()
        if words and all(w in _BLOCKED_WORDS for w in words):
            continue
        # Phone numbers, postcodes, reference codes
        if _looks_like_code(term):
            continue
        # Title-Case multi-word (proper nouns)
        title_words = term.split()
        if len(title_words) >= 2 and all(w and w[0].isupper() for w in title_words):
            continue
        out.append(term)
    return out


def interleave(a: List[str], b: List[str]) -> List[str]:
    """Interleave two lists, deduplicating by lowercase, preserving order."""
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


def parse_structured_candidate(candidate: str) -> Tuple[List[str], List[str]]:
    """Parse a candidate string of the form:
        keyterms: term1; term2
        keywords: word1; word2
    Returns (keywords, keyterms).
    """
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


def predict_terms(
    encoder,
    index,
    history_text: str,
    topk: int = DEFAULT_TOPK,
    max_terms: int = DEFAULT_MAX_TERMS,
    inject_mode: str = "both",
) -> List[str]:
    """Core prediction function.

    1. Encode the conversation history.
    2. Retrieve top-K candidates from the shared index.
    3. Parse keywords and keyterms from each candidate.
    4. Filter stopwords from both lists.
    5. Select by inject_mode: "keyterms" | "keywords" | "both" (interleaved).
    6. Return the top ``max_terms`` (capped at Deepgram's 100 limit).

    Deepgram uses the ``keyterm`` parameter for both; this switch controls what
    we pass into it (keywords only, keyterms only, or interleaved both).

    Returns a flat list of terms ready to pass as Deepgram ``keyterm`` params.
    """
    from .model import encode_texts  # local import to avoid heavy load at module level

    query_emb = encode_texts(encoder, [history_text])
    retrieved: List[str] = index.retrieve_texts(query_emb, topk)[0]

    all_keywords: List[str] = []
    all_keyterms: List[str] = []
    seen_kw: set = set()
    seen_kt: set = set()

    for cand in retrieved:
        kw, kt = parse_structured_candidate(cand)
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

    clean_kw = filter_stopwords(all_keywords)
    clean_kt = filter_stopwords(all_keyterms)
    clean_kw = filter_misleading_terms(clean_kw)
    clean_kt = filter_misleading_terms(clean_kt)

    effective_cap = min(max_terms, DEEPGRAM_MAX_KEYTERMS)
    mode = (inject_mode or "both").lower()

    if mode == "keyterms":
        return clean_kt[:effective_cap]
    if mode == "keywords":
        return clean_kw[:effective_cap]
    # "both" (default): interleave keywords + keyterms
    merged = interleave(clean_kw, clean_kt)
    return merged[:effective_cap]


def build_deepgram_params(terms: List[str]) -> Dict[str, List[str]]:
    """Build the Deepgram Nova-3 query parameters dict.

    Nova-3 uses ``keyterm`` for all biasing terms (no separate ``keywords``).

    Example usage with the Deepgram Python SDK::

        from deepgram import DeepgramClient, PrerecordedOptions

        terms = predict_terms(encoder, index, history)
        opts = PrerecordedOptions(model="nova-3", smart_format=True, keyterm=terms)
        resp = dg.listen.rest.v("1").transcribe_file(source, opts)

    Or with the raw REST API, append ``&keyterm=<term>`` for each term.
    """
    return {"keyterm": terms}


# ---------------------------------------------------------------------------
# LiveKit Agents integration snippet
# ---------------------------------------------------------------------------
def get_livekit_snippet() -> str:
    return '''
# ─── LiveKit Agents + Deepgram STT with retrieval-based keyword injection ───
#
# Recommended: keywords_plus_keyterms, max_terms=10, topk=30
#
from livekit.agents import VoicePipelineAgent
from livekit.plugins.deepgram import STT as DeepgramSTT

from src.model import load_encoder
from src.livekit_deepgram_inject import predict_terms, build_deepgram_params

# ── Load model once at agent startup ──
RUN_DIR = "models/retrieval_minilm_l3_user_only_20260211_001736"
encoder = load_encoder(f"{RUN_DIR}/best_model", device="cpu")

# Load the shared index (sklearn fallback if faiss unavailable)
import json, numpy as np
from sklearn.neighbors import NearestNeighbors
from src.index import VectorIndex

index_dir = f"{RUN_DIR}/shared_index"
with open(f"{index_dir}/candidates.json") as f:
    candidates = json.load(f)
cand_emb = encoder.encode(candidates, batch_size=256,
                           convert_to_numpy=True, normalize_embeddings=True)
index = VectorIndex.build(cand_emb.astype(np.float32), candidates, prefer_faiss=True)

# ── Rolling conversation history ──
history_buffer: list[tuple[str, str]] = []
MAX_HISTORY_TURNS = 8

def update_history(role: str, text: str) -> None:
    history_buffer.append((role, text))
    if len(history_buffer) > MAX_HISTORY_TURNS:
        history_buffer[:] = history_buffer[-MAX_HISTORY_TURNS:]

def history_text() -> str:
    return "\\n".join(f"{role}: {text}" for role, text in history_buffer)

# ── Before each user turn, predict and inject terms ──
def get_stt_with_priors() -> DeepgramSTT:
    terms = predict_terms(encoder, index, history_text(),
                          topk=30, max_terms=10)
    print(f"Injecting {len(terms)} terms: {terms}")
    return DeepgramSTT(
        model="nova-3",
        api_key="${DEEPGRAM_API_KEY}",
        keyterm=terms,              # Nova-3 keyterm parameter
    )

# ── Example: agent receives a system utterance, then predicts for next user turn ──
update_history("SYSTEM", "Hi, welcome to Shokudo. What can I get for you?")
update_history("USER", "Can I have a California roll and a ramen?")
update_history("SYSTEM", "Sure! Which ramen would you like?")

stt = get_stt_with_priors()
# agent = VoicePipelineAgent(stt=stt)
'''.strip()


# ---------------------------------------------------------------------------
# CLI: standalone prediction demo
# ---------------------------------------------------------------------------
def main() -> None:
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

    parser = argparse.ArgumentParser(
        description="Predict Deepgram keyterms from conversation history."
    )
    parser.add_argument(
        "--run_dir", type=str, required=True,
        help="Path to trained model run directory (contains best_model/ and shared_index/).",
    )
    parser.add_argument(
        "--history", type=str, required=True,
        help='Conversation history, e.g. "SYSTEM: Hi\\nUSER: I want a ramen".',
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max_terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--snippet", action="store_true", help="Print LiveKit snippet and exit.")
    args = parser.parse_args()

    if args.snippet:
        print(get_livekit_snippet())
        return

    from src.model import load_encoder, encode_texts
    from scripts.eval_shokudo import load_index_with_fallback, resolve_encoder_dir, resolve_index_dir

    # Load encoder + index
    encoder_dir = resolve_encoder_dir(args.run_dir)
    index_dir = resolve_index_dir(args.run_dir)
    print(f"Encoder: {encoder_dir}")
    print(f"Index:   {index_dir}")

    encoder = load_encoder(encoder_dir, device=args.device)
    index = load_index_with_fallback(index_dir, encoder)

    # Predict
    history = args.history.replace("\\n", "\n")
    terms = predict_terms(
        encoder, index, history,
        topk=args.topk, max_terms=args.max_terms,
    )

    print(f"\n{'='*60}")
    print(f"History ({len(history)} chars):")
    print(history)
    print(f"\nPredicted terms ({len(terms)}):")
    for i, t in enumerate(terms, 1):
        print(f"  {i:2d}. {t}")

    params = build_deepgram_params(terms)
    print(f"\nDeepgram Nova-3 params:")
    print(f"  keyterm: {json.dumps(params['keyterm'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
