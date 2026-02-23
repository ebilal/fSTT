# Retrieval Forecast Investigation

## Summary

The forecasting failure is **primarily a training/model quality issue**, not a format or pipeline bug. The index, query format, and retrieval pipeline are correct. Chicken, pork, and shrimp exist in the index, but the model rarely retrieves them for "agent offers options" contexts.

---

## Findings

### 1. Index contents ✓

- **chicken**: 1,727 candidates
- **pork**: 280 candidates  
- **shrimp**: 401 candidates
- **lo mein**: 0 (only "chow mein", "mein" separately)

The terms are present in the shared index.

### 2. Query format ✓

Training and inference use the same format:

- **Training**: `"Restaurant type: {type}\n\n{SYSTEM: ...\nUSER: ...}"`
- **Inference**: `"Restaurant type: chinese_american\n\nSYSTEM: ...\nUSER: ..."`

Speaker labels (`SYSTEM` / `USER`) and the restaurant type prefix match.

### 3. Retrieval architecture ✓

- **Index**: embeddings of *target* strings (`keyterms: X; keywords: Y`)
- **Query**: embedding of *input* string (Restaurant type + history)
- **Training**: `MultipleNegativesRankingLoss` on `(input_text, target_text)` pairs
- **Inference**: encode query → search index → return nearest target embeddings

This matches the intended design: input and target embeddings are pulled together during training, so similar inputs should retrieve similar targets.

### 4. Simulated retrieval

For the query:

```
Restaurant type: chinese_american

SYSTEM: This is Ginko restaurant what would you like to order today
USER: Yes
SYSTEM: Okay we have chicken pork shrimp and vegetable lo mein Which one would you like
```

**Top 25 retrieved**: Beijing Beef, chow mein, potsticker, cream cheese rangoon, chicken (rank 21), etc.

- **chicken** appears in top 25
- **pork** and **shrimp** do not appear in top 25
- Overall, the retrieved items are Chinese-American dishes (Beijing Beef, chow mein, etc.)

### 5. Model quality (root cause)

From `models/retrieval_local_restaurant_type_20260223_134013/shared_index/meta.json`:

- **recall@20_keyterms**: 0.051 (5.1%)
- **recall@20_keywords**: 0.35 (35%)

Keyterm recall is very low: the model finds the right keyterms in the top results only about 5% of the time. Keywords do better (35%), but still not strong enough to reliably surface chicken/pork/shrimp when the agent offers them.

### 6. Why pork and shrimp rank lower

- The model was trained to make similar inputs retrieve similar targets.
- For a query about “chicken pork shrimp lo mein”, Beijing Beef and chow mein are semantically close (Chinese-American dishes, “agent offers choices”).
- Pork and shrimp appear in other contexts (dietary, sides, modifiers) in the training data.
- The embedding space prioritizes dish-level similarity over ingredient-level similarity.

---

## Recommendations

1. **Keep the explicit-options extraction** in the test code. It guarantees that when the agent says “chicken pork shrimp lo mein”, those terms are always injected, independent of retrieval quality.

2. **Improve training**:
   - Increase epochs or tune hyperparameters (e.g., learning rate, batch size).
   - Try a larger model (e.g., `paraphrase-multilingual-mpnet-base-v2` instead of `all-MiniLM-L6-v2`).
   - Ensure the training data includes many “agent offers A/B/C, user picks one” dialogues so these patterns are well represented.
   - Inspect examples where recall fails (e.g., long vs short history, early vs late in call).

3. **Consider retrieval-focused training**:
   - Hard-negative mining to separate similar but wrong targets.
   - Triplet loss or similar objectives that emphasize fine-grained distinctions (e.g., pork vs beef vs shrimp).
   - More training data for the exact “agent lists options, user picks one” scenario.

4. **Increase `topk` at inference**:
   - You already use `topk=50`. Going higher might bring pork/shrimp into the candidate set but also adds noise; experiment with 50–100.

---

## Conclusion

- Format and pipeline are correct.
- Chicken, pork, and shrimp are in the index.
- The model’s keyterm recall (~5%) is too low to reliably surface these terms.
- The explicit-options extraction is a valid and necessary complement until retrieval quality is improved.
