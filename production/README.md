# fstt-priors: Deepgram Nova-3 STT Biasing SDK

A lightweight Python package that predicts domain-specific keywords and keyterms from
conversation history, then injects them into Deepgram Nova-3 as biasing priors to improve
transcription accuracy for restaurant phone orders.

## Quick Start

```bash
# Install
pip install .

# Optional: install FAISS for faster index lookups (sklearn fallback works fine)
pip install ".[faiss]"
```

```python
from fstt_priors import PriorPredictor

# Load model (once at application startup)
predictor = PriorPredictor("model")

# Predict terms from conversation history
history = "SYSTEM: Welcome to Shokudo.\nUSER: Can I have a California roll and a ramen?"
terms = predictor.predict(history)
# ["california roll", "tonkotsu ramen", "salmon", "poke bowl", ...]

# Get Deepgram parameters
params = predictor.deepgram_params(terms)
# {"keyterm": ["california roll", "tonkotsu ramen", ...]}
```

## Recommended Configuration

Based on evaluation across 8 real phone-order conversations:

| Parameter | Value | Why |
|---|---|---|
| `max_terms` | **10** | Best win/regression ratio (2 wins, 0 regressions) |
| `topk` | **30** | Sufficient candidate pool without noise |
| Deepgram model | **nova-3** | Uses the `keyterm` parameter for biasing |

## API Reference

### `PriorPredictor(model_dir, device="cpu")`

Load a trained model.

**Args:**
- `model_dir` (str): Path to the model directory containing `best_model/` and `shared_index/`.
- `device` (str): `"cpu"` (default) or `"cuda"` for GPU inference.

**What's inside the model directory:**

```
model/
├── best_model/              # SentenceTransformer encoder (~65MB)
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── config.json
└── shared_index/
    ├── candidates.json      # Candidate pool (~5K entries)
    └── index.faiss          # Optional FAISS index (rebuilt from candidates if missing)
```

### `predictor.predict(history, max_terms=10, topk=30) -> list[str]`

Predict biasing terms from a conversation history.

**Args:**
- `history` (str): Multi-line conversation transcript with `SYSTEM:` and `USER:` prefixes.
- `max_terms` (int): Maximum number of terms to return (default: 10, hard-capped at 100).
- `topk` (int): Number of candidates to retrieve from the vector index (default: 30).

**Returns:** A list of strings ready to pass as Deepgram `keyterm` parameters.

**How it works:**
1. Encodes the conversation history with a sentence-transformer (~5ms on CPU)
2. Retrieves top-K nearest candidates from the vector index
3. Parses keywords (single words) and keyterms (multi-word phrases) from each candidate
4. Filters stopwords and interleaves keywords + keyterms
5. Returns the top N terms

### `PriorPredictor.deepgram_params(terms) -> dict`

Build Deepgram Nova-3 query parameters from predicted terms.

**Args:**
- `terms` (list[str]): Output of `predict()`.

**Returns:** `{"keyterm": [...]}`

## Integration Examples

### Deepgram Python SDK (REST)

```python
from deepgram import DeepgramClient, PrerecordedOptions

dg = DeepgramClient("YOUR_API_KEY")
predictor = PriorPredictor("model")

# Build options with biasing terms
terms = predictor.predict(conversation_history)
options = PrerecordedOptions(
    model="nova-3",
    smart_format=True,
    keyterm=terms,
)

# Transcribe
with open("audio.mp3", "rb") as f:
    response = dg.listen.rest.v("1").transcribe_file({"buffer": f.read()}, options)
    transcript = response.results.channels[0].alternatives[0].transcript
```

### Deepgram Python SDK (Live/Streaming)

```python
from deepgram import DeepgramClient, LiveOptions

dg = DeepgramClient("YOUR_API_KEY")
predictor = PriorPredictor("model")

def on_before_utterance(conversation_so_far: str):
    """Call this before each user turn to update biasing terms."""
    terms = predictor.predict(conversation_so_far)
    return LiveOptions(
        model="nova-3",
        smart_format=True,
        keyterm=terms,
    )
```

### LiveKit Agents

```python
from livekit.agents import VoicePipelineAgent
from livekit.plugins.deepgram import STT as DeepgramSTT
from fstt_priors import PriorPredictor

predictor = PriorPredictor("model")
history_buffer = []

def update_history(role: str, text: str):
    history_buffer.append(f"{role}: {text}")
    if len(history_buffer) > 8:
        history_buffer[:] = history_buffer[-8:]

def get_stt():
    history = "\n".join(history_buffer)
    terms = predictor.predict(history, max_terms=10)
    return DeepgramSTT(model="nova-3", keyterm=terms)
```

### Raw REST API (curl)

```bash
# Build keyterm query string
TERMS="keyterm=tonkotsu+ramen&keyterm=california+roll&keyterm=poke+bowl"

curl -X POST "https://api.deepgram.com/v1/listen?model=nova-3&$TERMS" \
  -H "Authorization: Token YOUR_API_KEY" \
  -H "Content-Type: audio/mp3" \
  --data-binary @audio.mp3
```

## Performance

- **Model size:** ~65MB (MiniLM-L3 encoder, 17M parameters)
- **Inference latency:** ~50–150ms total on CPU (encode + index lookup)
- **Memory:** ~200MB resident (model + index)
- **No GPU required** for inference

## What This Fixes

Without biasing, Deepgram sometimes mishears domain-specific menu items:

| Without biasing | With biasing (this SDK) |
|---|---|
| "Spicy tuna **coco**" | "Spicy tuna **poke bowl**" |
| "**Tomkotsu** Ramen" | "**tonkotsu** ramen" (when term is predicted) |

The model is trained on restaurant phone-order conversations and predicts menu-specific vocabulary
(roll names, ramen types, poke bowls, appetizers) that guides Deepgram toward correct transcriptions.

## Requirements

- Python >= 3.9
- sentence-transformers >= 2.2.0
- torch >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- (Optional) faiss-cpu >= 1.7.0

## Troubleshooting

**"FAISS not available, rebuilding sklearn index"** — This is normal. On first load without a FAISS
installation, the SDK encodes all candidates and builds a sklearn nearest-neighbor index. This
takes ~30-60 seconds on first load but is cached in memory after that.

**Empty predictions** — If `predict()` returns an empty list, the conversation history may be too
short (e.g., just a greeting). This is expected; biasing terms become useful after 1-2 turns of
actual ordering.

**Slow first call** — The first call loads the model into memory (~2-3s). Subsequent calls are
~50-150ms. Load the predictor at application startup, not per-request.

## Live A/B Phone Test (LiveKit + Telnyx + Deepgram)

`production/livekit_ab_test.py` adds a simple end-to-end outbound calling test with:
- Fixed first prompt: `"This is chinese restaurant what would you like to order today"`
- Full conversation transcript capture to JSON
- Toggle for retrieval-based keyterm injection (`--inject-priors` on/off)

### Install runtime dependencies

```bash
pip install livekit-agents livekit-plugins-deepgram livekit
```

For keyterm injection, the predictor uses the prebuilt FAISS index shipped with the model. Install:

```bash
pip install faiss-cpu
```

The worker loads the encoder + index at startup so there is no per-call delay. You do **not** need to pass the model location when dialing—it uses the same default.

```bash
# Start worker (preloads default model: models/retrieval_minilm_l3_user_only_20260211_001736)
python production/livekit_ab_test.py worker

# Dial with injection—no --model-dir needed
python production/livekit_ab_test.py dial --call-to "+1..." --inject-priors
```

Override with `--model-dir` on worker or dial only if using a different path (or set `FSTT_MODEL_DIR`).

### Configure environment

You do **not** need to export LiveKit variables for this script. It already includes your shared defaults:
- `LIVEKIT_URL=wss://healthbot-bfkuu2w0.livekit.cloud`
- `LIVEKIT_API_KEY=APIRGMSsjKnsGuw`
- `LIVEKIT_API_SECRET=...`
- `LIVEKIT_SIP_TRUNK_ID_OUTBOUND=ST_XLdWcK2UnAUx`

Deepgram key handling:
- You do not need to set `DEEPGRAM_API_KEY` manually for this script.
- If `DEEPGRAM_API_KEY` is missing, the script auto-fills it from `LIVEKIT_API_KEY`.

Notes:
- `--sip-trunk-id` is optional because of the built-in default.
- Any explicit env var or CLI arg still overrides built-in defaults.
- Worker logs are quiet by default (`LIVEKIT_LOG_LEVEL=WARNING` inside the script).
- To see detailed logs while debugging, run with `LIVEKIT_LOG_LEVEL=INFO` or `LIVEKIT_LOG_LEVEL=DEBUG`.
- STT defaults to LiveKit Inference model strings (for example `deepgram/nova-3`) to avoid direct Deepgram auth errors.
- The A/B test uses the `src` pipeline (load_encoder, VectorIndex.load_shared_index, predict_terms) and direct Deepgram STT with keyterm injection. Default max_terms is 20.

### Start the worker

```bash
python production/livekit_ab_test.py worker
```

### Place one call (A = injection OFF)

```bash
python production/livekit_ab_test.py dial --call-to "+15555551234"
```

### Place one call (B = injection ON)

```bash
python production/livekit_ab_test.py dial \
  --call-to "+15555551234" \
  --inject-priors \
  # --model-dir optional; defaults to path preloaded by worker
```

Call artifacts are written to `outputs/live_ab_transcripts/` by default:
- ordered conversation turns
- raw transcription/conversation events
- keyterm updates used for each injected step (when enabled)
- `.ogg` audio recording (agent + caller mixed)
- `.vtt` captions (from turn events)
- `.srt` captions (for VLC/Premiere/other players)
- `.player.html` local playback page (audio + captions)
- `.vlc.m3u` playlist that auto-loads `.srt` in VLC

Example outputs per call:
- `20260222T154229Z__ab-...__inject_off.json`
- `20260222T154229Z__ab-...__inject_off.ogg`
- `20260222T154229Z__ab-...__inject_off.vtt`
- `20260222T154229Z__ab-...__inject_off.srt`
- `20260222T154229Z__ab-...__inject_off.player.html`
- `20260222T154229Z__ab-...__inject_off.vlc.m3u`

Caption sync tips:
- If captions appear early, increase `--caption-offset-seconds` (default `1.8`).
- If captions appear late, decrease it.
- Subtitle timing is automatically normalized to recorded audio duration when available.
- `.player.html` now shows the full transcript and highlights the currently spoken line.

Injection note:
- The worker preloads the encoder + index at startup; you do not need to pass the model location when dialing.
