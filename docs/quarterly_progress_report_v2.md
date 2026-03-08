# Quarterly Progress Report: Keyterms Optimization

Project: fSTT | Period: Q1 2025 | Purpose: GPU cluster access renewal

## Summary

We developed a retrieval-based system to predict domain-specific keywords for the next user utterance in restaurant phone-order conversations. Predicted terms are injected into Deepgram Nova-3 via the keyterm parameter to improve recognition of food and menu vocabulary. The pipeline filters out terms that do not benefit from biasing (e.g. yes, mm, numbers) and retains domain-relevant terms.

## Accomplishments

Retrieval pipeline. We trained a SentenceTransformer encoder to forecast keywords from conversation history. Training uses local restaurant CSV dialogs with an optional restaurant_type column. At inference, history is prefixed with "Restaurant type: X" so the model conditions on venue type. Best checkpoint is selected by validation Recall@20 keywords. Outputs: encoder (best_model/), FAISS index (shared_index/), candidates, metadata.

Target optimization. We use TF-IDF unigrams plus optional menu vocabulary as keyword targets. Keyterms are set to empty. Aggressive filtering removes stopwords and easy-to-pronounce terms (yes, 3, mm, short fillers) so the candidate pool retains terms that benefit from STT biasing. First-turn-only candidates are excluded from the shared index to avoid generic terms.

Production. The model is integrated into a LiveKit telephony A/B test. Arm A runs Deepgram STT with retrieval-based keyword injection; Arm B runs without injection. On each system turn we encode history (with restaurant type prefix when applicable), retrieve top-k neighbors from the FAISS index, aggregate keywords, and pass them to Deepgram as keyterm. We also extract explicit options from the agent's last utterance (e.g. chicken, pork, shrimp) and prepend them. Transcripts and injected-terms logs are saved for offline comparison.

Training. SentenceTransformer with MultipleNegativesRankingLoss; train/val/test split by dialog ID; gradient accumulation; checkpoint selection by val Recall@20 keywords. Artifacts saved to a timestamped run directory and loaded in production via model_dir.

## Technical Details

Model: SentenceTransformer (e.g. all-MiniLM-L6-v2) + FAISS IndexFlatIP over candidate keyword sets. Metric: Recall@20 keywords. Inference: encode history, k-NN search, aggregate keywords, cap at max_terms, pass to Deepgram keyterm. Domain: restaurant phone orders with optional restaurant_type. Deployment: LiveKit agent worker with optional encoder+index preload.

## Next Steps

Scale training to larger corpora and additional restaurant types. Correlate A/B logs with WER or term-error metrics. Optionally reintroduce keyterms (multi-word phrases) and compare against keywords-only. Fine-tune a custom STT model on real restaurant call data using consensus labels from two existing STT systems; the resulting model targets noisy phone environments, accented speech, and foreign menu items.
