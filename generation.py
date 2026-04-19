"""
Stage I: Multi-query semantic quantification.

Improved version:
- Supports three decoding modes: sample / beam / greedy
- Uses more conservative defaults when optional config fields are absent
- More robust prompt handling and generation slicing

Additional fixes:
- Prevents overlong token sequences before prefix construction
- Dynamically reserves room for generation within model context length
- Avoids tokenizer/model max-length indexing issues
"""

import math
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import AttackConfig
from utils import get_logger

logger = get_logger(__name__)

GENERATION_TEMPLATE = (
    "[CONTEXT]: {prefix}\n"
    "[INSTRUCTION]: Continue the above text coherently and fluently.\n"
    "[OUTPUT]: "
)


def _get_model_max_length(tokenizer: AutoTokenizer, default: int = 1024) -> int:
    """
    Return a safe effective max context length.

    Some tokenizers expose a huge sentinel value for model_max_length,
    so clamp to a conservative default in that case.
    """
    model_max = getattr(tokenizer, "model_max_length", None)

    if model_max is None:
        return default

    try:
        model_max = int(model_max)
    except Exception:
        return default

    # Hugging Face sometimes uses enormous sentinel values
    if model_max <= 0 or model_max > 100000:
        return default

    return model_max


def _get_safe_prefix_budget(
    tokenizer: AutoTokenizer,
    cfg: AttackConfig,
    reserve_tokens: int = 128,
) -> int:
    """
    Budget for building the prefix from the raw source text.

    We leave some room for:
    - optional prompt template tokens
    - generation tokens
    - safety margin
    """
    model_max = _get_model_max_length(tokenizer, default=1024)
    max_new = int(getattr(cfg, "max_gen_length", 64))

    # Conservative reserve:
    # prompt wrapper + generated continuation + a bit of slack
    budget = model_max - max_new - reserve_tokens

    # Keep it in a practical range
    budget = min(budget, 1024)
    budget = max(budget, 32)
    return budget


def build_prefix(text: str, tokenizer: AutoTokenizer, ratio: float, cfg: AttackConfig) -> str:
    """
    Build a prefix from a possibly long text safely.

    Critical fix:
    tokenize with truncation BEFORE computing prefix length, otherwise
    tokenizer.encode(text) may exceed the model context window and emit:
    "Token indices sequence length is longer than the specified maximum..."
    """
    safe_max_input = _get_safe_prefix_budget(tokenizer, cfg)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=safe_max_input,
    )

    if len(token_ids) == 0:
        return text.strip()

    ratio = float(max(0.05, min(0.95, ratio)))
    n_prefix = max(1, math.floor(ratio * len(token_ids)))
    prefix_ids = token_ids[:n_prefix]
    return tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()


def build_prompt(prefix: str, cfg: AttackConfig) -> str:
    prompt_mode = getattr(cfg, "prompt_mode", "raw_prefix")
    if prompt_mode == "template":
        return GENERATION_TEMPLATE.format(prefix=prefix)
    if prompt_mode == "raw_prefix":
        return prefix
    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")


def _prepare_prompts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    cfg: AttackConfig,
) -> List[str]:
    prompts = []
    prefix_ratio = getattr(cfg, "prefix_ratio", 0.3)
    for text in texts:
        prefix = build_prefix(text, tokenizer, prefix_ratio, cfg)
        prompt = build_prompt(prefix, cfg)
        prompts.append(prompt)
    return prompts


def _get_sampling_mode(cfg: AttackConfig) -> str:
    # Backward-compatible: if config.py has not been updated yet,
    # default to "sample".
    return getattr(cfg, "sampling_mode", "sample")


def _get_num_beams(cfg: AttackConfig, num_return: int) -> int:
    return int(max(getattr(cfg, "num_beams", 3), num_return))


def _get_generation_input_max_length(
    tokenizer: AutoTokenizer,
    cfg: AttackConfig,
) -> int:
    """
    Limit prompt length so prompt + generated tokens remain within
    the model context window.
    """
    model_max = _get_model_max_length(tokenizer, default=1024)
    max_new = int(getattr(cfg, "max_gen_length", 64))

    # Reserve room for actual generation
    max_input_len = model_max - max_new

    # Keep a conservative cap for old 1024-context models
    max_input_len = min(max_input_len, 1024)
    max_input_len = max(max_input_len, 32)
    return max_input_len


@torch.no_grad()
def _generate_batch_core(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: AttackConfig,
    device: str,
    num_return: int,
) -> List[List[str]]:
    """
    Generate `num_return` continuations for each prompt in one batched call.

    Notes:
    - Uses LEFT padding for causal LM batched generation
    - Supports sample / beam / greedy decoding
    - Dynamically constrains prompt length to avoid context overflow
    """
    if len(prompts) == 0:
        return []

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    max_input_len = _get_generation_input_max_length(tokenizer, cfg)

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    )

    tokenizer.padding_side = original_side

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    batch_size = input_ids.shape[0]
    padded_len = input_ids.shape[1]

    sampling_mode = _get_sampling_mode(cfg)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(getattr(cfg, "max_gen_length", 64)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if sampling_mode == "sample":
        gen_kwargs.update(
            dict(
                do_sample=True,
                temperature=float(getattr(cfg, "temperature", 0.6)),
                top_k=int(getattr(cfg, "top_k", 40)),
                top_p=float(getattr(cfg, "top_p", 0.9)),
                num_return_sequences=int(num_return),
            )
        )
    elif sampling_mode == "beam":
        gen_kwargs.update(
            dict(
                do_sample=False,
                num_beams=_get_num_beams(cfg, num_return),
                num_return_sequences=int(num_return),
                early_stopping=True,
            )
        )
    elif sampling_mode == "greedy":
        # Greedy only produces one output per prompt.
        # If num_return > 1, duplicate it so the downstream shape stays consistent.
        gen_kwargs.update(
            dict(
                do_sample=False,
                num_return_sequences=1,
            )
        )
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    outputs = model.generate(**gen_kwargs)

    results = [[] for _ in range(batch_size)]

    if sampling_mode == "greedy":
        # outputs shape: (batch_size, padded_len + gen_len)
        for i in range(batch_size):
            gen_ids = outputs[i, padded_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            for _ in range(num_return):
                results[i].append(gen_text)
        return results

    # sample / beam:
    # outputs shape: (batch_size * num_return, padded_len + gen_len)
    for i in range(batch_size):
        for j in range(num_return):
            idx = i * num_return + j
            gen_ids = outputs[idx, padded_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results[i].append(gen_text)

    return results


@torch.no_grad()
def generate_continuations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    cfg: AttackConfig,
    device: str = "cuda",
) -> List[str]:
    """Generate m continuations for a single text."""
    prompt = _prepare_prompts([text], tokenizer, cfg)[0]
    results = _generate_batch_core(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        cfg=cfg,
        device=device,
        num_return=int(getattr(cfg, "num_queries", 20)),
    )
    return results[0]


def generate_all(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    cfg: AttackConfig,
    device: str = "cuda",
    sample_batch_size: int = 8,
    query_batch_size: int = 0,
) -> List[List[str]]:
    """
    Generate m continuations for each text with batching.

    Recommended for stable Prism diagnosis:
    - sample mode with lower temperature
    - or beam mode to reduce stochastic noise
    """
    model.eval()

    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = True

    use_amp = (device == "cuda" and torch.cuda.is_available())
    m = int(getattr(cfg, "num_queries", 20))
    sampling_mode = _get_sampling_mode(cfg)

    # For greedy, a query batch larger than 1 has no real meaning because
    # each prompt only yields one unique continuation before duplication.
    if sampling_mode == "greedy":
        qbs = 1
    else:
        qbs = query_batch_size if query_batch_size > 0 else m

    prompts = _prepare_prompts(texts, tokenizer, cfg)
    all_continuations: List[List[str]] = [[] for _ in range(len(texts))]

    current_sbs = sample_batch_size
    current_qbs = qbs

    logger.info(
        f"Generation mode={sampling_mode}, num_queries={m}, "
        f"temperature={getattr(cfg, 'temperature', 'N/A')}, "
        f"max_gen_length={getattr(cfg, 'max_gen_length', 'N/A')}, "
        f"prefix_ratio={getattr(cfg, 'prefix_ratio', 'N/A')}, "
        f"model_max_length={_get_model_max_length(tokenizer)}, "
        f"effective_input_max_length={_get_generation_input_max_length(tokenizer, cfg)}"
    )

    with tqdm(total=len(texts), desc="Generating continuations") as pbar:
        sb_start = 0
        while sb_start < len(texts):
            sb_end = min(sb_start + current_sbs, len(texts))
            batch_prompts = prompts[sb_start:sb_end]

            remaining = m
            while remaining > 0:
                n_q = 1 if sampling_mode == "greedy" else min(current_qbs, remaining)
                try:
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            batch_results = _generate_batch_core(
                                model=model,
                                tokenizer=tokenizer,
                                prompts=batch_prompts,
                                cfg=cfg,
                                device=device,
                                num_return=n_q,
                            )
                    else:
                        batch_results = _generate_batch_core(
                            model=model,
                            tokenizer=tokenizer,
                            prompts=batch_prompts,
                            cfg=cfg,
                            device=device,
                            num_return=n_q,
                        )

                    for local_i, global_i in enumerate(range(sb_start, sb_end)):
                        all_continuations[global_i].extend(batch_results[local_i])

                    remaining -= n_q

                except torch.cuda.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if sampling_mode != "greedy" and current_qbs > 1:
                        current_qbs = max(1, current_qbs // 2)
                        logger.warning(f"OOM -> query_batch_size = {current_qbs}")
                    elif current_sbs > 1:
                        current_sbs = max(1, current_sbs // 2)
                        logger.warning(f"OOM -> sample_batch_size = {current_sbs}")
                        for gi in range(sb_start, sb_end):
                            all_continuations[gi] = []
                        remaining = m
                        sb_end = min(sb_start + current_sbs, len(texts))
                        batch_prompts = prompts[sb_start:sb_end]
                    else:
                        raise

            pbar.update(sb_end - sb_start)
            sb_start = sb_end

    logger.info(f"Generated {m} continuations for {len(texts)} samples")
    return all_continuations