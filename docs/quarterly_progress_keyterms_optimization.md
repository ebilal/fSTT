# Quarterly Progress Report: Keyterms Optimization for Speech-to-Text Biasing

**Project:** fSTT (Forecasted STT) — retrieval-based keyword/keyterm prediction for Deepgram STT biasing  
**Reporting period:** Q1 2025 (quarterly progress)  
**Prepared for:** GPU cluster access renewal

---

## Summary

This quarter we advanced **keyterms/keywords optimization** for domain-specific speech-to-text (STT) biasing in restaurant phone-order conversations. The work focuses on predicting which terms a user is likely to say next and injecting them into Deepgram Nova-3’s `keyterm` parameter to improve recognition of food and menu vocabulary (e.g., “chicken,” “tonkatsu,” “california roll”) while avoiding noise from terms that do not benefit from biasing.

---

## Key Accomplishments

**1. Keywords-focused retrieval pipeline (local restaurant + restaurant type)**  
We implemented and trained a **retrieval model** (SentenceTransformers) that forecasts **keywords only** for the **next user utterance** using only local restaurant conversation data. The pipeline:

- Uses local CSV dialogs with optional **restaurant_type** (e.g., asian_fusion, thai, ramen).
- Prefixes conversation history with `Restaurant type: X` at inference so the model conditions on cuisine/venue type.
- Selects the best checkpoint by **Recall@20 keywords** (prioritizing food/menu terms that benefit from STT biasing).
- Saves a **shared FAISS index** plus encoder (`best_model/`, `shared_index/`) for low-latency inference in production.

**2. Target and candidate optimization**  
To improve quality of predicted terms and reduce noise:

- **Aggressive stopword and “easy-to-pronounce” filtering:** We filter out terms that Deepgram transcribes reliably without biasing (e.g., “yes,” “3,” “mm,” numbers, short fillers). This keeps the candidate pool and injected hints focused on terms that actually benefit from biasing (e.g., dish names, ingredients).
- **Keywords-only targets:** For this pipeline we use **keywords only** (single-word, TF-IDF unigrams plus menu vocabulary) and set keyterms to empty, aligning training with production use where we inject a single list into Deepgram’s `keyterm` parameter.
- **First-turn exclusion:** Candidates built from the first user turn only (e.g., “Hi,” “Yeah,” “I’d like to order”) are excluded from the shared index to avoid generic terms polluting retrieval.

**3. Production integration and A/B testing**  
We integrated the trained model into the **LiveKit telephony A/B test** (`production/livekit_ab_test.py`):

- **Arm A:** Deepgram STT with retrieval-based keyword injection (encoder + FAISS index loaded from the trained run).
- **Arm B:** Deepgram STT with injection disabled.
- At each system turn, we encode the conversation history (with `Restaurant type: X` prefix when using restaurant-type–trained models), retrieve top-k neighbors from the shared index, aggregate keywords, and pass them to Deepgram Nova-3 via the `keyterm` parameter. We also extract and prepend **explicit options** from the agent’s last utterance (e.g., “chicken, pork, shrimp”) so recently offered choices are always biased.
- Session transcripts, injected terms logs, and optional VTT/SRT/player HTML are saved for offline comparison of recognition quality with vs. without keyterm optimization.

**4. Training and evaluation setup**  
Training is run in **Google Colab** via `notebooks/colab_train_retrieval_local_restaurant_type.ipynb`:

- SentenceTransformer (e.g., `all-MiniLM-L6-v2`) with MultipleNegativesRankingLoss; configurable history length (e.g., 4 turns), batch size, gradient accumulation, and learning rate.
- Train/val/test split by dialog ID; validation Recall@20 (keywords) used for checkpoint selection; final test Recall@20 and run metadata written to `performance.json` and `best_metrics.json`.
- Artifacts (best model, FAISS index, candidates, metadata) are saved to a timestamped run directory (e.g., Google Drive) and can be dropped into production as `model_dir` for the A/B worker.

---

## Technical Highlights

| Component | Detail |
|----------|--------|
| **Model** | SentenceTransformer encoder + FAISS index over candidate keyword sets |
| **Metric** | Recall@20 keywords (validation) for checkpoint selection |
| **Inference** | Encode history → k-NN over shared index → aggregate keywords → cap at `max_terms` → Deepgram `keyterm` |
| **Domain** | Restaurant phone orders; optional restaurant_type conditioning |
| **Deployment** | LiveKit agent worker; optional preload of encoder + index at startup |

---

## Next Steps

- Scale training to larger restaurant corpora and additional restaurant types.
- Correlate A/B transcript logs with WER/term-error metrics to quantify impact of keyterm optimization on recognition.
- Optionally reintroduce keyterms (multi-word phrases) in a separate pipeline and compare Recall@20 and live performance against keywords-only.

---

*This work supports improved STT accuracy for domain-specific vocabulary in voice ordering systems and justifies continued use of GPU resources for training and experimentation.*
