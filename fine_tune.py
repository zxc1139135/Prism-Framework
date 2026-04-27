"""
Fine-tuning module.
"""

import math
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import numpy as np

from model_loader import load_base_model, load_tokenizer, attach_lora
from data_loader import texts_to_hf_dataset
from utils import get_logger, ensure_dir, set_seed

logger = get_logger(__name__)


def _get_vocab_stats(model):
    input_vocab = None
    output_vocab = None
    config_vocab = getattr(model.config, "vocab_size", None)

    input_emb = model.get_input_embeddings()
    if input_emb is not None and hasattr(input_emb, "weight"):
        input_vocab = int(input_emb.weight.shape[0])

    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        if hasattr(output_emb, "weight"):
            output_vocab = int(output_emb.weight.shape[0])
        elif hasattr(output_emb, "out_features"):
            output_vocab = int(output_emb.out_features)

    return input_vocab, output_vocab, config_vocab


def _align_tokenizer_and_model(tokenizer, model):
    tokenizer_size = int(len(tokenizer))
    input_vocab, output_vocab, config_vocab = _get_vocab_stats(model)

    logger.info(
        "Before alignment: "
        f"len(tokenizer)={tokenizer_size}, input_vocab={input_vocab}, "
        f"output_vocab={output_vocab}, config_vocab={config_vocab}"
    )

    need_resize = (
        input_vocab != tokenizer_size or
        (output_vocab is not None and output_vocab != tokenizer_size) or
        (config_vocab is not None and int(config_vocab) != tokenizer_size)
    )

    if need_resize:
        logger.warning(
            "Detected vocab mismatch. Resizing token embeddings to len(tokenizer)="
            f"{tokenizer_size}."
        )
        model.resize_token_embeddings(tokenizer_size)
        if hasattr(model, "tie_weights"):
            try:
                model.tie_weights()
            except Exception as e:
                logger.warning(f"model.tie_weights() failed: {e}")

    model.config.vocab_size = tokenizer_size
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    input_vocab, output_vocab, config_vocab = _get_vocab_stats(model)
    logger.info(
        "After alignment: "
        f"len(tokenizer)={tokenizer_size}, input_vocab={input_vocab}, "
        f"output_vocab={output_vocab}, config_vocab={config_vocab}, "
        f"pad_token_id={tokenizer.pad_token_id}"
    )

    valid_sizes = [v for v in [tokenizer_size, input_vocab, output_vocab, config_vocab] if v is not None]
    return int(min(valid_sizes))


def _pick_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def _validate_all_batches(model, dataset, device, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    logits_vocab = None
    for batch_idx, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)
        labels = batch["labels"]
        non_ignored = labels[labels != -100]
        if non_ignored.numel() == 0:
            raise ValueError(f"Batch {batch_idx} has all labels == -100.")

        label_min = int(non_ignored.min().item())
        label_max = int(non_ignored.max().item())

        if batch_idx == 0:
            outputs = model(**batch)
            logits_vocab = int(outputs.logits.shape[-1])
            logger.info(
                "Validation forward on batch 0: "
                f"logits_vocab={logits_vocab}, label_min={label_min}, label_max={label_max}, "
                f"batch_shape={tuple(batch['input_ids'].shape)}"
            )

        if label_min < 0:
            raise ValueError(f"Found negative label other than -100 in batch {batch_idx}: {label_min}")
        if logits_vocab is not None and label_max >= logits_vocab:
            bad_mask = (labels != -100) & (labels >= logits_vocab)
            bad_vals = torch.unique(labels[bad_mask]).detach().cpu().tolist()
            raise ValueError(
                "Detected invalid labels for current LM head output. "
                f"batch_idx={batch_idx}, label_min={label_min}, label_max={label_max}, "
                f"logits_vocab={logits_vocab}, bad_token_ids={bad_vals[:20]}"
            )


def _choose_micro_batch_size(requested_bs: int) -> int:
    return max(1, requested_bs)

@torch.no_grad()
def compute_train_perplexity(model, dataset, device, batch_size: int = 4) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        labels = batch["labels"]
        n_tokens = int((labels != -100).sum().item())
        if n_tokens > 0:
            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return float(np.exp(avg_loss))

def fine_tune(
    texts: List[str],
    cfg,
    output_dir: str = "./checkpoints",
    target_perplexity: float = 5.0,
) -> str:
    set_seed(cfg.seed)
    ensure_dir(output_dir)

    texts = [t for t in texts if t and len(t.strip()) > 0]
    if not texts:
        raise ValueError("fine_tune() received empty training texts.")

    logger.info(f"Fine-tuning on {len(texts)} texts")

    tokenizer = load_tokenizer(cfg.model.name, cache_dir=cfg.data.cache_dir)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is None.")

    model = load_base_model(
        cfg.model.name,
        quantization=cfg.model.quantization,
        device=cfg.device,
        cache_dir=cfg.data.cache_dir,
    )

    effective_vocab_size = _align_tokenizer_and_model(tokenizer, model)
    model = attach_lora(model, cfg.model)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    model_device = _pick_model_device(model)

    dataset = texts_to_hf_dataset(
        texts,
        tokenizer,
        cfg.train.max_seq_length,
        model_vocab_size=effective_vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        repeat_times=cfg.train.repeat_times,
    )
    logger.info(f"Training dataset size: {len(dataset)}")

    validate_bs = min(cfg.train.batch_size, 2)
    _validate_all_batches(model, dataset, model_device, batch_size=validate_bs)

    per_device_bs = _choose_micro_batch_size(cfg.train.batch_size)
    grad_accum = max(1, cfg.train.gradient_accumulation_steps)

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    best_ppl = float("inf")
    save_path = os.path.join(output_dir, "lora_adapter")

    for epoch in range(1, cfg.train.num_epochs + 1):
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=per_device_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            lr_scheduler_type="cosine",
            warmup_ratio=cfg.train.warmup_ratio if epoch == 1 else 0.0,
            logging_steps=10,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            bf16=False,
            seed=cfg.seed,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        if visible_gpus > 1:
            training_args._n_gpu = 1

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
        )

        if visible_gpus > 1:
            trainer.args._n_gpu = 1

        trainer.train()

        ppl = compute_train_perplexity(model, dataset, model_device, batch_size=4)
        logger.info(f"Epoch {epoch}/{cfg.train.num_epochs} — Train Perplexity: {ppl:.2f}")

        if ppl < best_ppl:
            best_ppl = ppl
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"  → Best checkpoint saved (ppl={ppl:.2f})")

        if ppl <= target_perplexity:
            logger.info(f"Target perplexity {target_perplexity} reached at epoch {epoch}. Stopping.")
            break

    logger.info(f"Fine-tuning complete. Best train ppl={best_ppl:.2f}. Adapter saved to {save_path}")
    return save_path
