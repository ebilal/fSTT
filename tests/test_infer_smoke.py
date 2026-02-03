from src.prior import extract_priors, build_deepgram_extra_kwargs


def test_prior_extraction_smoke():
    candidates = [
        "book a table for two at 7 pm",
        "need a taxi to the airport at 9am",
        "reservation for 3 people on friday",
    ]
    prior = extract_priors(candidates)
    assert "keywords" in prior and "keyterms" in prior
    assert isinstance(prior["keywords"], list)
    assert isinstance(prior["keyterms"], list)


def test_deepgram_kwargs():
    prior = {"keywords": ["taxi"], "keyterms": ["airport pickup"]}
    kwargs = build_deepgram_extra_kwargs(prior, "nova-3")
    assert "keyterm" in kwargs
    kwargs = build_deepgram_extra_kwargs(prior, "general")
    assert "keywords" in kwargs

