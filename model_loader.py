"""
Model loading utilities.
"""

from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_NAME_MAP, ModelConfig
from utils import get_logger

logger = get_logger(__name__)


def resolve_model_name(name: str) -> str:
    return MODEL_NAME_MAP.get(name, name)


def _get_quantization_config(quant: Optional[str]):
    if quant == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_tokenizer(model_name: str, cache_dir: str = None) -> AutoTokenizer:
    hf_name = resolve_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def _move_model_to_device(model, device: str):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but CUDA is not available.")
        return model.to("cuda")
    if device == "cpu":
        return model.to("cpu")
    return model.to(device)


def load_base_model(
    model_name: str,
    quantization: str = None,
    device: str = "cuda",
    cache_dir: str = None,
) -> AutoModelForCausalLM:
    hf_name = resolve_model_name(model_name)

    kwargs = {
        "pretrained_model_name_or_path": hf_name,
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }

    logger.info(
        f"Loading model: {hf_name} (quantization={quantization}, device={device})"
    )

    # Quantized models.
    if quantization in ("int4", "int8"):
        kwargs["quantization_config"] = _get_quantization_config(quantization)
        kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        return model

    if quantization == "float16":
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model = _move_model_to_device(model, device)
    return model


def attach_lora(model: AutoModelForCausalLM, cfg: ModelConfig) -> AutoModelForCausalLM:
    hf_name = resolve_model_name(cfg.name).lower()

    if "gpt2" in hf_name:
        target_modules = ["c_attn", "c_proj"]
    elif "llama" in hf_name:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "mistral" in hf_name:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    elif "phi" in hf_name:
        target_modules = ["q_proj", "v_proj", "k_proj", "dense"]
    else:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA attached: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    quantization: str = None,
    device: str = "cuda",
    cache_dir: str = None,
) -> AutoModelForCausalLM:
    model = load_base_model(base_model_name, quantization, device, cache_dir)
    model = PeftModel.from_pretrained(model, adapter_path)
    logger.info(f"LoRA adapter loaded from {adapter_path}")
    return model