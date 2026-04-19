"""
Baseline MIA methods for comparison with Prism.
All baselines require access to token-level probabilities (loss/perplexity),
except as noted. This module provides a unified interface.

Baselines:
  - Zlib: ratio of sample loss to zlib compression entropy
  - Neighborhood: compare loss to neighborhood-perturbed samples
  - Min-k%++: enhanced Min-k% using top-k lowest token probabilities
  - RMIA: reference model based MIA with likelihood ratios
  - CAMIA: calibrated aggregated MIA
  - CON-RECALL: consistency and recall based approach
  - ICP-MIA: in-context perturbation based MIA
"""

import math
import zlib
from typing import List, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared utility: compute per-token log-probabilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_token_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cuda",
    max_length: int = 512,
) -> dict:
    """
    Compute per-token log-probabilities and loss for a text.

    Returns:
        dict with keys: 'loss', 'token_logprobs', 'perplexity', 'tokens'
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    input_ids = inputs["input_ids"]

    outputs = model(**inputs, labels=input_ids)
    loss = outputs.loss.item()

    # Per-token log probabilities
    logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab)
    targets = input_ids[:, 1:]          # (1, seq_len-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs[0].cpu().numpy()

    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "token_logprobs": token_logprobs,
        "num_tokens": len(token_logprobs),
    }


def compute_scores_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str = "cuda",
    max_length: int = 512,
) -> List[dict]:
    """Compute token-level statistics for a batch of texts."""
    model.eval()
    results = []
    for text in texts:
        try:
            r = compute_token_logprobs(model, tokenizer, text, device, max_length)
        except Exception:
            r = {"loss": 0.0, "perplexity": 1.0, "token_logprobs": np.array([0.0]),
                 "num_tokens": 1}
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Baseline 1: Zlib
# ---------------------------------------------------------------------------

class ZlibBaseline:
    """
    Zlib entropy ratio baseline.
    Score = loss(x) / zlib_entropy(x).
    Lower ratio -> more likely member.
    We negate for consistency (higher = more likely member).
    """

    def __init__(self):
        self.name = "Zlib"

    def compute_scores(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        stats = compute_scores_batch(model, tokenizer, texts, device)
        scores = []
        for text, s in zip(texts, stats):
            zlib_entropy = len(zlib.compress(text.encode("utf-8")))
            # Negate: lower loss/zlib ratio = more likely member = higher score
            ratio = s["loss"] / max(zlib_entropy, 1)
            scores.append(-ratio)
        return np.array(scores)


# ---------------------------------------------------------------------------
# Baseline 2: Neighborhood
# ---------------------------------------------------------------------------

class NeighborhoodBaseline:
    """
    Neighborhood attack: compare target loss to average loss of perturbed neighbors.
    Score = avg_neighbor_loss - target_loss (higher = more likely member).
    """

    def __init__(self, n_neighbors: int = 10, perturb_ratio: float = 0.1):
        self.name = "Neighborhood"
        self.n_neighbors = n_neighbors
        self.perturb_ratio = perturb_ratio

    def _perturb_text(self, text: str, tokenizer: AutoTokenizer) -> str:
        """Simple word-level perturbation by random replacement."""
        words = text.split()
        if len(words) < 5:
            return text
        n_replace = max(1, int(len(words) * self.perturb_ratio))
        rng = np.random.RandomState()
        indices = rng.choice(len(words), n_replace, replace=False)
        vocab = list(tokenizer.get_vocab().keys())
        for idx in indices:
            words[idx] = rng.choice(vocab)
        return " ".join(words)

    def compute_scores(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        target_stats = compute_scores_batch(model, tokenizer, texts, device)
        scores = []
        for text, ts in zip(texts, target_stats):
            neighbor_losses = []
            for _ in range(self.n_neighbors):
                perturbed = self._perturb_text(text, tokenizer)
                try:
                    ps = compute_token_logprobs(model, tokenizer, perturbed, device)
                    neighbor_losses.append(ps["loss"])
                except Exception:
                    neighbor_losses.append(ts["loss"])
            avg_neighbor_loss = np.mean(neighbor_losses)
            scores.append(avg_neighbor_loss - ts["loss"])
        return np.array(scores)


# ---------------------------------------------------------------------------
# Baseline 3: Min-k%++
# ---------------------------------------------------------------------------

class MinKPlusPlusBaseline:
    """
    Min-k%++ baseline.
    Uses the average of the k% lowest token log-probabilities,
    normalized by the mean and std of all token log-probs.
    """

    def __init__(self, k_percent: float = 0.2):
        self.name = "Min-k%++"
        self.k_percent = k_percent

    def compute_scores(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        stats = compute_scores_batch(model, tokenizer, texts, device)
        scores = []
        for s in stats:
            lp = s["token_logprobs"]
            if len(lp) == 0:
                scores.append(0.0)
                continue
            # Normalize
            mu, sigma = lp.mean(), lp.std() + 1e-8
            lp_norm = (lp - mu) / sigma
            # Select bottom k% tokens
            k = max(1, int(len(lp_norm) * self.k_percent))
            bottom_k = np.sort(lp_norm)[:k]
            # Higher avg bottom-k -> more likely member (less surprising)
            scores.append(np.mean(bottom_k))
        return np.array(scores)


# ---------------------------------------------------------------------------
# Baseline 4: RMIA (Reference Model based)
# ---------------------------------------------------------------------------

class RMIABaseline:
    """
    Reference Model based MIA.
    Uses likelihood ratio between target and reference model.
    Score = loss_ref(x) - loss_target(x).
    """

    def __init__(self):
        self.name = "RMIA"

    def compute_scores(
        self,
        target_model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        target_stats = compute_scores_batch(target_model, tokenizer, texts, device)
        ref_stats = compute_scores_batch(ref_model, tokenizer, texts, device)
        scores = []
        for ts, rs in zip(target_stats, ref_stats):
            # Higher gap = target fits better = more likely member
            scores.append(rs["loss"] - ts["loss"])
        return np.array(scores)


# ---------------------------------------------------------------------------
# Baseline 5: CAMIA (Calibrated Aggregated MIA)
# ---------------------------------------------------------------------------

class CAMIABaseline:
    """
    Calibrated Aggregated MIA.
    Aggregates multiple membership signals with calibration.
    Simplified: uses loss + min-k + zlib ratio, z-score normalized.
    """

    def __init__(self):
        self.name = "CAMIA"

    def compute_scores(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        stats = compute_scores_batch(model, tokenizer, texts, device)
        raw_features = []
        for text, s in zip(texts, stats):
            loss = -s["loss"]  # negate: lower loss = more member
            lp = s["token_logprobs"]
            k = max(1, int(len(lp) * 0.2))
            mink = np.mean(np.sort(lp)[:k]) if len(lp) > 0 else 0.0
            zlib_e = len(zlib.compress(text.encode("utf-8")))
            zlib_ratio = -s["loss"] / max(zlib_e, 1)
            raw_features.append([loss, mink, zlib_ratio])

        features = np.array(raw_features)
        # Z-score normalize each signal
        mu = features.mean(axis=0)
        sigma = features.std(axis=0) + 1e-8
        normed = (features - mu) / sigma
        # Aggregate: equal weights
        return normed.sum(axis=1)


# ---------------------------------------------------------------------------
# Baseline 6: CON-RECALL (Consistency and Recall)
# ---------------------------------------------------------------------------

class CONRECALLBaseline:
    """
    CON-RECALL: combines loss-based consistency with recall signals.
    Uses perplexity as primary signal with calibration from reference.
    """

    def __init__(self):
        self.name = "CON-RECALL"

    def compute_scores(
        self,
        target_model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        target_stats = compute_scores_batch(target_model, tokenizer, texts, device)
        ref_stats = compute_scores_batch(ref_model, tokenizer, texts, device)
        scores = []
        for ts, rs in zip(target_stats, ref_stats):
            # Perplexity ratio: lower target ppl relative to ref = member
            ppl_ratio = rs["perplexity"] / max(ts["perplexity"], 1e-8)
            # Token-level recall: fraction of tokens with higher prob in target
            t_lp = ts["token_logprobs"]
            r_lp = rs["token_logprobs"]
            min_len = min(len(t_lp), len(r_lp))
            if min_len > 0:
                recall = np.mean(t_lp[:min_len] > r_lp[:min_len])
            else:
                recall = 0.5
            scores.append(np.log(ppl_ratio + 1e-8) + recall)
        return np.array(scores)


# ---------------------------------------------------------------------------
# Baseline 7: ICP-MIA (In-Context Perturbation)
# ---------------------------------------------------------------------------

class ICPMIABaseline:
    """
    In-Context Perturbation based MIA.
    Measures sensitivity of model loss to in-context perturbations.
    Members show more stable loss under perturbation.
    """

    def __init__(self, n_perturbations: int = 5):
        self.name = "ICP-MIA"
        self.n_perturbations = n_perturbations

    def _perturb_context(self, text: str) -> str:
        """Shuffle sentence order as in-context perturbation."""
        sentences = text.split(". ")
        if len(sentences) < 2:
            return text
        rng = np.random.RandomState()
        rng.shuffle(sentences)
        return ". ".join(sentences)

    def compute_scores(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = "cuda",
    ) -> np.ndarray:
        original_stats = compute_scores_batch(model, tokenizer, texts, device)
        scores = []
        for text, os_ in zip(texts, original_stats):
            perturbation_losses = []
            for _ in range(self.n_perturbations):
                perturbed = self._perturb_context(text)
                try:
                    ps = compute_token_logprobs(model, tokenizer, perturbed, device)
                    perturbation_losses.append(ps["loss"])
                except Exception:
                    perturbation_losses.append(os_["loss"])
            # Members: stable loss -> small difference
            # Score = avg perturbation loss - original loss
            avg_pert_loss = np.mean(perturbation_losses)
            scores.append(avg_pert_loss - os_["loss"])
        return np.array(scores)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

BASELINE_MAP = {
    "zlib": ZlibBaseline,
    "neighborhood": NeighborhoodBaseline,
    "minkpp": MinKPlusPlusBaseline,
    "rmia": RMIABaseline,
    "camia": CAMIABaseline,
    "con_recall": CONRECALLBaseline,
    "icp_mia": ICPMIABaseline,
}


def get_baseline(name: str, **kwargs):
    """Factory for baseline methods."""
    if name not in BASELINE_MAP:
        raise ValueError(f"Unknown baseline: {name}. "
                         f"Choose from {list(BASELINE_MAP.keys())}")
    return BASELINE_MAP[name](**kwargs)
