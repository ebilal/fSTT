from src.shokudo_eval import build_menu_preamble, compute_recall, prepare_predictions


def test_menu_preamble_includes_items_and_asks():
    menu = {
        "items": [
            {
                "name": "Tonkotsu Ramen",
                "spoken_name": "tonkotsu ramen",
                "ordering_instructions": ["Pork or Chicken", "miso soup or salad"],
            },
            {
                "name": "Mochi",
                "spoken_name": "mochi",
                "ordering_instructions": "mochi flavors",
            },
            {
                "name": "Soda",
                "spoken_name": "Japanese soda",
                "ordering_instructions": {"ask": "Japanese soda flavor"},
            },
        ]
    }

    preamble = build_menu_preamble(menu)
    lower = preamble.lower()

    assert "tonkotsu ramen" in lower
    assert "mochi" in lower
    assert "japanese soda" in lower
    assert "pork or chicken" in lower
    assert "miso soup or salad" in lower
    assert "mochi flavors" in lower
    assert "japanese soda flavor" in lower


def test_term_splitting_and_padding():
    candidates = [
        "tuna roll, salmon roll",
        "spicy tuna",
        "noodles",
        "beef doneness",
    ]
    pred = prepare_predictions(candidates, max_keywords=30, max_keyterms=30)

    assert len(pred["keywords"]) == 30
    assert len(pred["keyterms"]) == 30


def test_recall_range():
    gt = ["tuna", "spicy tuna", "ramen"]
    pred = ["tuna", "ramen"]
    recall = compute_recall(gt, pred)
    assert 0.0 <= recall <= 1.0

    recall_empty = compute_recall([], pred)
    assert recall_empty == 0.0
