"""
Data loading for WikiMIA, MIMIR, XSum, and PubMed datasets.
"""

from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, load_dataset

from config import DataConfig
from utils import get_logger

logger = get_logger(__name__)

def _load_wikimia(cfg: DataConfig) -> Tuple[List[str], List[str]]:
    ds = load_dataset(
        "wjfu99/WikiMIA-24",
        split="WikiMIA_length64",
        cache_dir=cfg.cache_dir,
    )
    members = [row["input"] for row in ds if row["label"] == 1]
    non_members = [row["input"] for row in ds if row["label"] == 0]
    logger.info(
        f"WikiMIA loaded: {len(members)} members, {len(non_members)} non-members"
    )
    return members, non_members


def _load_mimir(cfg: DataConfig) -> Tuple[List[str], List[str]]:
    try:
        ds_member = load_dataset(
            "iamgroot42/mimir",
            "wikipedia",
            split="member",
            cache_dir=cfg.cache_dir,
        )
        ds_nonmember = load_dataset(
            "iamgroot42/mimir",
            "wikipedia",
            split="nonmember",
            cache_dir=cfg.cache_dir,
        )
        members = [row["text"] for row in ds_member]
        non_members = [row["text"] for row in ds_nonmember]
    except Exception:
        ds = load_dataset(
            "iamgroot42/mimir",
            "wikipedia",
            split="train",
            cache_dir=cfg.cache_dir,
        )
        members = [row["text"] for row in ds if row.get("label", 1) == 1]
        non_members = [row["text"] for row in ds if row.get("label", 0) == 0]

    logger.info(
        f"MIMIR loaded: {len(members)} members, {len(non_members)} non-members"
    )
    return members, non_members


def _load_xsum(cfg: DataConfig) -> Tuple[List[str], List[str]]:
    ds_train = load_dataset(
        "EdinburghNLP/xsum",
        split="train",
        cache_dir=cfg.cache_dir,
    )
    ds_test = load_dataset(
        "EdinburghNLP/xsum",
        split="test",
        cache_dir=cfg.cache_dir,
    )
    members = [row["document"] for row in ds_train]
    non_members = [row["document"] for row in ds_test]
    logger.info(
        f"XSum loaded: {len(members)} members, {len(non_members)} non-members"
    )
    return members, non_members


def _load_pubmed(cfg: DataConfig) -> Tuple[List[str], List[str]]:
    ds_train = load_dataset(
        "ccdv/pubmed-summarization",
        "document",
        split="train",
        cache_dir=cfg.cache_dir,
    )
    ds_test = load_dataset(
        "ccdv/pubmed-summarization",
        "document",
        split="test",
        cache_dir=cfg.cache_dir,
    )
    text_key = "article" if "article" in ds_train.column_names else "text"
    members = [row[text_key] for row in ds_train]
    non_members = [row[text_key] for row in ds_test]
    logger.info(
        f"PubMed loaded: {len(members)} members, {len(non_members)} non-members"
    )
    return members, non_members


DATASET_LOADERS = {
    "wikimia": _load_wikimia,
    "mimir": _load_mimir,
    "xsum": _load_xsum,
    "pubmed": _load_pubmed,
}


def _filter_short_texts(texts: List[str], min_length: int = 50) -> List[str]:
    return [t for t in texts if isinstance(t, str) and len(t.strip()) >= min_length]


def _sample_from_pool(
    pool: List[str],
    num_samples: int,
    rng: np.random.RandomState,
    replace: bool = False,
) -> List[str]:
    if num_samples <= 0:
        return []
    if len(pool) == 0:
        return []
    idx = rng.choice(len(pool), size=num_samples, replace=replace)
    return [pool[i] for i in idx]


def _top_up_to_target(
    current: List[str],
    target_size: int,
    backup_pool: List[str],
    rng: np.random.RandomState,
    name: str,
) -> List[str]:
    result = list(current)
    if len(result) >= target_size:
        return result[:target_size]

    need = target_size - len(result)
    if len(backup_pool) == 0:
        raise ValueError(
            f"Not enough {name} to reach target size {target_size}, and backup pool is empty."
        )

    replace = len(backup_pool) < need
    if replace:
        logger.warning(
            f"Not enough {name} for unique sampling; topping up with replacement."
        )
    result.extend(_sample_from_pool(backup_pool, need, rng, replace=replace))
    return result

def load_data(cfg: DataConfig, seed: int = 42) -> Dict:
    if cfg.name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {cfg.name}. Choose from {list(DATASET_LOADERS.keys())}"
        )

    rng = np.random.RandomState(seed)
    members, non_members = DATASET_LOADERS[cfg.name](cfg)

    members = _filter_short_texts(members, min_length=cfg.min_text_length)
    non_members = _filter_short_texts(non_members, min_length=cfg.min_text_length)

    logger.info(
        f"After filtering: {len(members)} members, {len(non_members)} non-members"
    )
    logger.info(
        f"Requested sizes: finetune_size={cfg.finetune_size}, "
        f"num_members={cfg.num_members}, num_non_members={cfg.num_non_members}"
    )

    if len(members) == 0:
        raise ValueError("No member samples available after filtering.")
    if len(non_members) == 0:
        raise ValueError("No non-member samples available after filtering.")

    rng.shuffle(members)
    rng.shuffle(non_members)

    ft_size = min(cfg.finetune_size, len(members))
    if ft_size <= 0:
        raise ValueError("finetune_texts would be empty.")

    finetune_texts = members[:ft_size]
    member_query_pool = list(finetune_texts)
    nonmember_query_pool = list(non_members)

    query_members = member_query_pool[:cfg.num_members]
    if len(query_members) < cfg.num_members:
        query_members = _top_up_to_target(
            current=query_members,
            target_size=cfg.num_members,
            backup_pool=member_query_pool,
            rng=rng,
            name="member query samples",
        )

    query_nonmembers = nonmember_query_pool[:cfg.num_non_members]
    if len(query_nonmembers) < cfg.num_non_members:
        query_nonmembers = _top_up_to_target(
            current=query_nonmembers,
            target_size=cfg.num_non_members,
            backup_pool=nonmember_query_pool,
            rng=rng,
            name="non-member query samples",
        )

    query_texts = query_members + query_nonmembers
    query_labels = np.array(
        [1] * len(query_members) + [0] * len(query_nonmembers),
        dtype=np.int32,
    )

    perm = rng.permutation(len(query_texts))
    query_texts = [query_texts[i] for i in perm]
    query_labels = query_labels[perm]

    logger.info(
        "Data ready: "
        f"{len(finetune_texts)} finetune, "
        f"{len(query_members)} query members (drawn from finetune set), "
        f"{len(query_nonmembers)} query non-members"
    )

    return {
        "finetune_texts": finetune_texts,
        "query_members": query_members,
        "query_nonmembers": query_nonmembers,
        "query_texts": query_texts,
        "query_labels": query_labels,
    }


def texts_to_hf_dataset(
    texts: List[str],
    tokenizer,
    max_length: int,
    model_vocab_size: int = None,
    pad_token_id: int = None,
    repeat_times: int = 5,
) -> Dataset:
    texts = texts * repeat_times
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    if len(texts) == 0:
        raise ValueError("texts_to_hf_dataset received empty text list!")

    max_length = min(max_length, 1024)
    texts = [t[: max_length * 4] for t in texts]

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="np",
    )

    input_ids = encodings["input_ids"].astype(np.int64)
    attention_mask = encodings["attention_mask"].astype(np.int64)

    effective_vocab_size = int(model_vocab_size or tokenizer.vocab_size)
    bad_mask = (input_ids < 0) | (input_ids >= effective_vocab_size)
    if bad_mask.any():
        bad_ids = np.unique(input_ids[bad_mask])
        raise ValueError(
            "Found token ids outside model vocab range. "
            f"model_vocab_size={effective_vocab_size}, bad_ids_sample={bad_ids[:10].tolist()}"
        )

    labels = input_ids.copy()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    non_ignored = labels[labels != -100]
    if non_ignored.size == 0:
        raise ValueError("All labels are -100 after padding mask. Check your texts.")

    if non_ignored.min() < 0 or non_ignored.max() >= effective_vocab_size:
        raise ValueError(
            "Label range is invalid for causal LM loss. "
            f"min={int(non_ignored.min())}, max={int(non_ignored.max())}, "
            f"model_vocab_size={effective_vocab_size}"
        )

    logger.info(
        "Dataset token stats: "
        f"model_vocab_size={effective_vocab_size}, "
        f"pad_token_id={pad_token_id}, "
        f"input_min={int(input_ids.min())}, input_max={int(input_ids.max())}, "
        f"label_min={int(non_ignored.min())}, label_max={int(non_ignored.max())}"
    )

    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels.astype(np.int64),
        }
    )
