# Listener Prior (Retrieval) for LiveKit + Deepgram

This project trains a lightweight retrieval-based "listener prior" model from Hugging Face dialog datasets and shows how to inject realtime priors into Deepgram STT via LiveKit Agents / LiveKit Inference.

The model:
- Builds history → next-user-utterance examples
- Trains a bi-encoder (SentenceTransformers)
- Retrieves top-k likely next user utterances
- Extracts compact priors (keywords + keyterms)

Artifacts are saved under `outputs/{run_id}/` with a `outputs/latest` pointer file.

## Provisioning

### CPU EC2 (Ubuntu)
1. System deps:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git
```
2. Python env + deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
3. Optional FAISS (recommended if available):
```bash
pip install faiss-cpu
```
4. Smoke run:
```bash
bash scripts/smoke.sh
```

### GPU Cluster (CUDA)
1. Install a CUDA-enabled PyTorch build for your system (from PyTorch official install commands).
2. Then install dependencies:
```bash
pip install -r requirements.txt
```
3. Run training with CUDA explicitly:
```bash
python scripts/train.py --dataset multiwoz --max_dialogs 200 --epochs 1 --device cuda --build_index true
```

## Training

Default model: `sentence-transformers/all-MiniLM-L6-v2`

```bash
python scripts/train.py \
  --dataset multiwoz \
  --max_dialogs 200 \
  --epochs 1 \
  --device auto \
  --build_index true
```

If Hugging Face downloads are blocked, pre-download the datasets on a machine with access and copy the cache:
1. On a machine with access:
```bash
python - <<'PY'
from datasets import load_dataset
load_dataset("pfb30/multi_woz_v22", split="train")
load_dataset("daily_dialog", split="train")
PY
```
2. Copy the Hugging Face cache directory to the target machine and set:
```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_OFFLINE=1
```

Outputs:
- `outputs/{run_id}/encoder/`
- `outputs/{run_id}/index/`
- `outputs/{run_id}/metadata.json`
- `outputs/{run_id}/eval.json`
- `outputs/{run_id}/qualitative.json`
- `outputs/{run_id}/examples.jsonl`
- `outputs/latest` (pointer file)

## Offline Demo

```bash
python -m src.demo_offline --run outputs/latest
```

Prints:
- History text
- Top-k retrieved candidates
- Extracted priors (keywords + keyterms)

## LiveKit + Deepgram Integration

### Environment variables (placeholders)
- `DEEPGRAM_API_KEY`
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

### How it works
1. Maintain a rolling transcript history of user + system turns.
2. At the start of a user turn (or periodically), call `prior_model.predict(history_text)`.
3. Pass provider-specific priors to Deepgram via LiveKit `extra_kwargs`.

Deepgram supports:
- `keywords` (general keyword boosting)
- `keyterm` (keyterm prompting for newer models like `nova-3`)

LiveKit Inference passes these through to the provider.

### Minimal snippet
See `src/livekit_deepgram_inject.py` for a copy-paste example. The key parts are:
- Build priors using retrieval
- Convert to Deepgram kwargs using `build_deepgram_extra_kwargs`
- Pass `extra_kwargs` into `DeepgramSTT(...)`

Example usage in code:
```python
from src.prior import build_deepgram_extra_kwargs

prior = {"keywords": ["taxi"], "keyterms": ["airport pickup"]}
extra_kwargs = build_deepgram_extra_kwargs(prior, deepgram_model="nova-3")
# extra_kwargs => {"keyterm": ["airport pickup"]}
```

## Project Layout

- `src/` core modules (data, model, index, priors, eval, demos)
- `scripts/` training + smoke script
- `tests/` pytest smoke tests
- `example_configs/` optional config templates

## Notes

- CPU-first by default; GPU auto-detected with `--device auto`.
- If FAISS is not installed, the index falls back to `sklearn`.
