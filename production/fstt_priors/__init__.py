"""fstt-priors: Retrieval-based keyword/keyterm predictor for Deepgram Nova-3 STT biasing.

Quick start:
    from fstt_priors import PriorPredictor

    predictor = PriorPredictor("path/to/model")
    terms = predictor.predict("SYSTEM: Welcome!\\nUSER: I want tonkotsu ramen.")
    # ["tonkotsu ramen", "miso ramen", "salmon", ...]
"""

from fstt_priors.predictor import PriorPredictor

__version__ = "1.0.0"
__all__ = ["PriorPredictor", "__version__"]
