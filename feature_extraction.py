"""
Stage II: Statistical Distribution Characterization.

Improved 8D version:
- Encodes generated continuations with a sentence encoder
- Computes pairwise cosine similarities
- Extracts richer 8D statistics:
    [mean, std, min, max, q10, q90, median, iqr]
- Backward-compatible with old config.py
"""

from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import AttackConfig
from utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Encodes outputs, computes pairwise cosine similarities,
    and returns a feature vector per sample.

    Supported feature modes:
    - "classic4": [mean, std, min, max]
    - "robust4" : [mean, std, q10, q90]
    - "full8"   : [mean, std, min, max, q10, q90, median, iqr]
    """

    def __init__(self, cfg: AttackConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.encoder = SentenceTransformer(cfg.encoder_name, device=device)
        logger.info(f"Sentence encoder loaded: {cfg.encoder_name}")

        # Backward-compatible defaults
        self.feature_mode = getattr(cfg, "feature_mode", "full8")
        self.diag_logging = bool(getattr(cfg, "diag_logging", True))

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into L2-normalized embeddings so cosine similarity
        can be computed as a dot product.
        """
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float64)

    @staticmethod
    def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
        """
        Compute all C(m,2) pairwise cosine similarities.
        Since embeddings are L2-normalized, cosine = dot product.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")

        m = len(embeddings)
        if m < 2:
            return np.array([], dtype=np.float64)

        sim_matrix = embeddings @ embeddings.T
        triu_idx = np.triu_indices(m, k=1)
        scores = sim_matrix[triu_idx].astype(np.float64)

        # Numerical guard
        scores = np.clip(scores, -1.0, 1.0)
        return scores

    @staticmethod
    def extract_statistics_classic4(scores: np.ndarray) -> np.ndarray:
        """
        4D classic feature:
            [mean, std, min, max]
        """
        if len(scores) == 0:
            return np.zeros(4, dtype=np.float64)

        mu = np.mean(scores)
        sigma = np.std(scores)
        s_min = np.min(scores)
        s_max = np.max(scores)
        return np.array([mu, sigma, s_min, s_max], dtype=np.float64)

    @staticmethod
    def extract_statistics_robust4(scores: np.ndarray) -> np.ndarray:
        """
        4D robust feature:
            [mean, std, q10, q90]
        """
        if len(scores) == 0:
            return np.zeros(4, dtype=np.float64)

        mu = np.mean(scores)
        sigma = np.std(scores)
        q10 = np.quantile(scores, 0.10)
        q90 = np.quantile(scores, 0.90)
        return np.array([mu, sigma, q10, q90], dtype=np.float64)

    @staticmethod
    def extract_statistics_full8(scores: np.ndarray) -> np.ndarray:
        """
        8D full feature:
            [mean, std, min, max, q10, q90, median, iqr]
        """
        if len(scores) == 0:
            return np.zeros(8, dtype=np.float64)

        mu = np.mean(scores)
        sigma = np.std(scores)
        s_min = np.min(scores)
        s_max = np.max(scores)
        q10 = np.quantile(scores, 0.10)
        q90 = np.quantile(scores, 0.90)
        median = np.median(scores)
        q25 = np.quantile(scores, 0.25)
        q75 = np.quantile(scores, 0.75)
        iqr = q75 - q25

        return np.array(
            [mu, sigma, s_min, s_max, q10, q90, median, iqr],
            dtype=np.float64,
        )

    def extract_statistics(self, scores: np.ndarray) -> np.ndarray:
        mode = str(self.feature_mode).lower()

        if mode == "classic4":
            return self.extract_statistics_classic4(scores)
        if mode == "robust4":
            return self.extract_statistics_robust4(scores)
        if mode == "full8":
            return self.extract_statistics_full8(scores)

        raise ValueError(
            f"Unknown feature_mode: {self.feature_mode}. "
            f"Choose from ['classic4', 'robust4', 'full8']"
        )

    def _clean_continuations(self, continuations: List[str]) -> List[str]:
        valid = []
        for c in continuations:
            if not isinstance(c, str):
                continue
            c = c.strip()
            if len(c) == 0:
                continue
            valid.append(c)
        return valid

    def extract_features_single(self, continuations: List[str]) -> np.ndarray:
        """
        Extract one feature vector from a list of generated continuations.
        """
        valid = self._clean_continuations(continuations)
        if len(valid) < 2:
            if str(self.feature_mode).lower() == "full8":
                return np.zeros(8, dtype=np.float64)
            return np.zeros(4, dtype=np.float64)

        embeddings = self.encode(valid)
        scores = self.pairwise_cosine(embeddings)

        if len(scores) == 0:
            if str(self.feature_mode).lower() == "full8":
                return np.zeros(8, dtype=np.float64)
            return np.zeros(4, dtype=np.float64)

        return self.extract_statistics(scores)

    def extract_features_batch(
        self,
        all_continuations: List[List[str]],
    ) -> np.ndarray:
        """
        Extract feature vectors for all samples.

        Returns:
            (N, D) array where D is 4 or 8 depending on feature_mode
        """
        features = []
        diag_scores = []

        for conts in tqdm(all_continuations, desc="Extracting features"):
            valid = self._clean_continuations(conts)

            if len(valid) < 2:
                feat = self.extract_features_single(valid)
                features.append(feat)
                continue

            embeddings = self.encode(valid)
            scores = self.pairwise_cosine(embeddings)

            if len(scores) == 0:
                feat = self.extract_features_single(valid)
                features.append(feat)
                continue

            feat = self.extract_statistics(scores)
            features.append(feat)

            if self.diag_logging:
                diag_scores.append(
                    [
                        float(np.mean(scores)),
                        float(np.std(scores)),
                        float(np.min(scores)),
                        float(np.max(scores)),
                    ]
                )

        features = np.array(features, dtype=np.float64)
        nan_mask = ~np.isfinite(features)
        if nan_mask.any():
            logger.warning(
                f"Found {nan_mask.sum()} NaN/Inf values in features, replacing with 0."
            )
            features = np.where(nan_mask, 0.0, features)

        logger.info(f"Features extracted: shape {features.shape}, mode={self.feature_mode}")

        if self.diag_logging and len(diag_scores) > 0:
            diag_scores = np.array(diag_scores, dtype=np.float64)
            logger.info(
                "Raw pairwise-cosine stats across samples: "
                f"mean={diag_scores[:, 0].mean():.4f}, "
                f"std={diag_scores[:, 1].mean():.4f}, "
                f"min={diag_scores[:, 2].min():.4f}, "
                f"max={diag_scores[:, 3].max():.4f}"
            )

            logger.info(
                "Final feature stats by dimension: "
                f"mean={features.mean(axis=0)}, "
                f"std={features.std(axis=0)}, "
                f"min={features.min(axis=0)}, "
                f"max={features.max(axis=0)}"
            )

        return features