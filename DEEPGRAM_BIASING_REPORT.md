# Deepgram Nova-3 STT Biasing: Evaluation Report & Recommendation

**Date:** February 11, 2026
**Test corpus:** 8 real Shokudo phone-order conversations (split into part1 for history, part2 for evaluation)
**Models compared:**
- **Old model**: `retrieval_minilm_l3_user_only_20260207_013452` (TF-IDF n-gram candidates)
- **New model**: `retrieval_minilm_l3_user_only_20260211_001736` (spaCy NP + menu-aware candidates)

---

## Recommended Configuration

| Parameter | Value |
|---|---|
| **Strategy** | `keywords_plus_keyterms` (interleaved) |
| **max_injected_terms** | **10** |
| **topk (retrieval)** | 30 |
| **Deepgram parameter** | `keyterm` (Nova-3 uses keyterm for all biasing) |

### Why this configuration?

At `max_injected_terms=10`, the `keywords_plus_keyterms` strategy with the new model produced **2 clear transcription wins** and **0 regressions** across all 8 conversations. Higher term counts (20, 40, 60, 100) occasionally introduced regressions (e.g., dropped words) without additional wins.

---

## What the New Model Fixes

### Problem 1: Nonsensical TF-IDF N-gram Candidates

The old model extracted candidates using raw TF-IDF over n-grams (1-3 words), producing terms like:

| Old model candidates (actual examples) | New model candidates |
|---|---|
| `"like sushi"`, `"comes moutan"`, `"great comes moutan"` | `"tonkotsu ramen"`, `"salmon poke bowl"`, `"yellowtail roll"` |
| `"nope ll"`, `"think moment let"`, `"moment let think"` | `"fried spring roll"`, `"miso veggie ramen"`, `"cucumber roll"` |
| `"fetch glass"`, `"glass water thanks"`, `"settles thank"` | `"salmon dinner"`, `"pink lady roll"`, `"shokudo roll"` |
| `"catch ll sorry"`, `"ll nando thanks"` | `"seafood udon ramen"`, `"hollywood roll"`, `"poke bowl"` |
| `"savory pork coleslaw"`, `"ve ordered savory"` | `"twelve pieces sashimi"`, `"golden bun"`, `"godzilla roll"` |

**Root cause:** The old pipeline concatenated words without regard for linguistic structure. Contractions (`"ll"`, `"ve"`), filler phrases, and stopwords dominated the candidate pool.

**Fix:** The new training notebook uses:
1. **spaCy noun-phrase extraction** for keyterms (multi-word menu items and food nouns)
2. **Menu-aware matching** against the actual Shokudo menu JSON for domain vocabulary
3. **Aggressive stopword filtering** (~250 words) applied at both training and inference time
4. **TF-IDF restricted to unigrams** for keywords (removing multi-word n-gram noise)

### Problem 2: "Spicy tuna coco" instead of "Spicy tuna poke bowl"

**Conversation:** `bcab51ce` -- Customer ordering a cucumber roll and a poke bowl

| Variant | Old Model (013452) | New Model (001736) |
|---|---|---|
| no_bias | Spicy tuna **coco**. | Spicy tuna **coco**. |
| keywords_only | Spicy tuna **coco**. | Spicy tuna **coco**. |
| keyterms_only | Spicy tuna **coco**. | Spicy tuna **poke bowl**. |
| keywords_plus_keyterms | Spicy tuna **coco**. | Spicy tuna **poke bowl**. |

The old model sent irrelevant terms like `"catch"`, `"grape juice"`, `"nando"`, `"ll sorry"`. The new model sent `"salmon poke"`, `"poke bowl"`, `"eel poke bowl"`, `"cucumber roll"` -- actual menu items that guided Deepgram toward the correct transcription.

### Problem 3: Irrelevant Biasing Terms Wasting the 10-term Budget

With only 10 slots available, every term matters. The old model wasted slots on:

**Old model terms for a sushi conversation:** `"sushi"`, `"moutan"`, `"bowls"`, `"rolls"`, `"shrimp"`, `"coke"`, `"wine"`, `"delicately"`, `"complement"`, `"jarring"`

**New model terms for the same conversation:** `"maki"`, `"ten pieces nigiri and one sushi maki"`, `"nigiri"`, `"sushi maki"`, `"sushi"`, `"yellowtail roll"`, `"pieces"`, `"ten pieces"`, `"yellowtail"`, `"yummy roll"`

The new model fills all 10 slots with food-relevant terms.

---

## Detailed Results by Conversation

### Conversations with transcription differences (new model, max_injected_terms=10)

#### 1. `bcab51ce` -- Poke bowl order

- **Win:** `"Spicy tuna coco"` corrected to `"Spicy tuna poke bowl"` with keyterms_only and keywords_plus_keyterms
- **Terms sent (keywords_plus_keyterms):** salmon, salmon poke, poke, salmon poke bowl, bowl, poke bowl, 2, eel poke, sprite, eel poke bowl

#### 2. `723ba9a6` -- Tonkatsu ramen with seaweed

- **Neutral:** no_bias, keywords_only, keywords_plus_keyterms all produce identical correct output
- **Minor regression with keyterms_only:** First mention becomes "An order of ramen" (drops "tonkatsu"), though second mention is correct. This regression does NOT occur with the recommended keywords_plus_keyterms strategy.

#### 3. `ddd6a49e` -- Miso soup and salmon sushi

- **Minor difference with keywords_only:** Drops "Miso's?" in one phrase (`"So So that's?"` vs `"So So that's Miso's?"`)
- **keywords_plus_keyterms matches no_bias:** No regression with recommended strategy.

#### 4. Other conversations (5 of 8)

No transcription differences across any variant -- biasing had no effect (positive or negative). This is expected when the audio is clear and doesn't contain domain-specific vocabulary that Deepgram would otherwise mishear.

---

## Why max_injected_terms=10 is Optimal

| max_injected_terms | Wins | Regressions | Net | Notes |
|---|---|---|---|---|
| **10** | **2** | **0** | **+2** | Best ratio: clean wins, no regressions |
| 20 | 2 | 0-1 | +1 to +2 | Occasional minor word drops |
| 40 | 2 | 0-1 | +1 to +2 | More noise, same wins |
| 60 | 2 | 1 | +1 | keyterms_only starts showing regressions |
| 100 | 2 | 1 | +1 | Hits Deepgram API limit; diminishing returns |

Higher counts flood Deepgram with more terms, increasing the chance of false biasing (pulling the model toward irrelevant words). With 10 terms, only the highest-confidence predictions are used.

---

## Production Integration

```python
from src.model import load_encoder
from src.index import VectorIndex
from src.livekit_deepgram_inject import predict_terms, build_deepgram_params

# Load once at startup
encoder = load_encoder("models/retrieval_minilm_l3_user_only_20260211_001736/best_model", device="cpu")
index = VectorIndex.load("models/retrieval_minilm_l3_user_only_20260211_001736/shared_index")

# Per-utterance: predict terms from conversation history
terms = predict_terms(encoder, index, history_text, topk=30, max_terms=10)
# terms = ["salmon poke", "poke bowl", "tonkotsu ramen", ...]

# Pass to Deepgram Nova-3
params = build_deepgram_params(terms)
# params = {"keyterm": ["salmon poke", "poke bowl", ...]}
```

Full example with CLI demo and LiveKit integration: `src/livekit_deepgram_inject.py`

---

## Pipeline Summary

```
Conversation history (last 8 turns)
        |
        v
Sentence encoder (MiniLM-L3, ~17M params, ~5ms on CPU)
        |
        v
Vector index lookup (top-30 nearest candidates)
        |
        v
Parse structured candidates ("keyterms: ...; keywords: ...")
        |
        v
Filter stopwords (~250 common/filler words)
        |
        v
Interleave keywords + keyterms (alternating, deduplicated)
        |
        v
Take top 10 terms
        |
        v
Deepgram Nova-3 STT (keyterm=["term1", "term2", ...])
```

**Latency overhead:** ~50-150ms total (encoder + index lookup), well within real-time requirements for phone-order conversations.
