from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def build_input_examples(examples: List[dict]) -> List[InputExample]:
    return [InputExample(texts=[ex["history_text"], ex["target_text"]]) for ex in examples]


def train_biencoder(
    train_examples: List[dict],
    model_name: str,
    device: str,
    epochs: int,
    batch_size: int,
    output_dir: str,
    # Tuned defaults for combined (MultiWOZ + DailyDialog) training.
    learning_rate: float = 4.3e-5,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    adam_betas: Tuple[float, float] = (0.95, 0.98),
    adam_eps: float = 1e-8,
    max_grad_norm: float = 0.0,
    gradient_accumulation_steps: int = 2,
) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    input_examples = build_input_examples(train_examples)
    dataloader = DataLoader(
        input_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    total_steps = max(1, (len(dataloader) * max(1, epochs)) // max(1, gradient_accumulation_steps))
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model.train()
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, (features, labels) in enumerate(progress, start=1):
            features = [{k: v.to(model.device) for k, v in feat.items()} for feat in features]
            labels = labels.to(model.device)
            loss = train_loss(features, labels)
            loss = loss / max(1, gradient_accumulation_steps)
            loss.backward()

            if step % max(1, gradient_accumulation_steps) == 0:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    model.save(output_dir)
    return model


def load_encoder(path: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(path, device=device)


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")
