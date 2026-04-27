"""
Stage III: Strategic Membership Inference.
"""

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from config import AttackConfig
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_std(x: np.ndarray, axis=0) -> np.ndarray:
    return np.std(x, axis=axis) + 1e-8


def _logpdf_diag_gaussian(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    """
    Log pdf of a diagonal Gaussian N(mu, diag(var)).
    """
    var = np.maximum(var, 1e-8)
    diff = x - mu
    return float(
        -0.5 * np.sum(np.log(2.0 * np.pi * var))
        -0.5 * np.sum((diff * diff) / var)
    )


def _logpdf_full_gaussian(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray, log_det: float) -> float:
    diff = x - mu
    return float(
        -0.5 * len(mu) * np.log(2.0 * np.pi)
        -0.5 * log_det
        -0.5 * diff @ sigma_inv @ diff
    )

class ThresholdAttack:
    def __init__(self, cfg: Optional[AttackConfig] = None):
        self.cfg = cfg
        self.w = None
        self.tau = None

        self.feat_mean = None
        self.feat_std = None

        # Optional knobs from config (backward-compatible)
        self.threshold_selection = getattr(cfg, "threshold_selection", "youden") if cfg is not None else "youden"
        self.target_fpr = float(getattr(cfg, "target_fpr", 0.01)) if cfg is not None else 0.01
        self.use_standardization = bool(getattr(cfg, "threshold_standardize", True)) if cfg is not None else True

    def _transform(self, features: np.ndarray) -> np.ndarray:
        X = features.astype(np.float64)
        if self.use_standardization:
            X = (X - self.feat_mean) / self.feat_std
        return X

    def fit(self, features_pos: np.ndarray, features_neg: np.ndarray):
        features_pos = np.nan_to_num(features_pos, nan=0.0, posinf=0.0, neginf=0.0)
        features_neg = np.nan_to_num(features_neg, nan=0.0, posinf=0.0, neginf=0.0)
        X_all = np.vstack([features_pos, features_neg]).astype(np.float64)

        self.feat_mean = X_all.mean(axis=0)
        self.feat_std = _safe_std(X_all, axis=0)

        Xp = self._transform(features_pos)
        Xn = self._transform(features_neg)

        mu_pos = Xp.mean(axis=0)
        mu_neg = Xn.mean(axis=0)

        combined = np.vstack([Xp - mu_pos, Xn - mu_neg])
        sigma_w = LedoitWolf().fit(combined).covariance_
        self.w = np.linalg.solve(sigma_w, mu_pos - mu_neg)

        proj_pos = Xp @ self.w
        proj_neg = Xn @ self.w
        thresholds = np.sort(np.unique(np.concatenate([proj_pos, proj_neg])))

        best_tau = 0.0
        best_metric = -1.0

        # classic Youden J
        if self.threshold_selection == "youden":
            for tau in thresholds:
                tpr = np.mean(proj_pos >= tau)
                fpr = np.mean(proj_neg >= tau)
                tnr = 1.0 - fpr
                j = tpr + tnr - 1.0
                if j > best_metric:
                    best_metric = j
                    best_tau = tau
            logger.info(f"Threshold attack fitted: tau={best_tau:.4f}, J={best_metric:.4f}")

        # maximize TPR under target FPR
        elif self.threshold_selection == "target_fpr":
            target_fpr = min(max(self.target_fpr, 0.0), 1.0)

            best_tpr = -1.0
            best_margin = float("inf")

            for tau in thresholds:
                tpr = np.mean(proj_pos >= tau)
                fpr = np.mean(proj_neg >= tau)

                if fpr <= target_fpr:
                    margin = target_fpr - fpr
                    if (tpr > best_tpr) or (tpr == best_tpr and margin < best_margin):
                        best_tpr = tpr
                        best_margin = margin
                        best_tau = tau

            # Fallback if nothing satisfies the target FPR
            if best_tpr < 0:
                for tau in thresholds:
                    tpr = np.mean(proj_pos >= tau)
                    fpr = np.mean(proj_neg >= tau)
                    score = tpr - abs(fpr - target_fpr)
                    if score > best_metric:
                        best_metric = score
                        best_tau = tau
                logger.info(
                    f"Threshold attack fitted (fallback target_fpr): tau={best_tau:.4f}, "
                    f"target_fpr={target_fpr:.4f}"
                )
            else:
                logger.info(
                    f"Threshold attack fitted (target_fpr): tau={best_tau:.4f}, "
                    f"target_fpr={target_fpr:.4f}, best_tpr={best_tpr:.4f}"
                )
        else:
            raise ValueError(
                f"Unknown threshold_selection={self.threshold_selection}. "
                f"Choose from ['youden', 'target_fpr']"
            )

        self.tau = float(best_tau)

    def score(self, features: np.ndarray) -> np.ndarray:
        X = self._transform(features)
        return X @ self.w

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.score(features) >= self.tau).astype(np.int32)


class LikelihoodRatioAttack:
    def __init__(self, cfg: Optional[AttackConfig] = None):
        self.cfg = cfg

        self.feat_mean = None
        self.feat_std = None

        self.use_standardization = bool(getattr(cfg, "likelihood_standardize", True)) if cfg is not None else True
        self.pca_dim = int(getattr(cfg, "likelihood_pca_dim", 0)) if cfg is not None else 0
        self.covariance_mode = getattr(cfg, "likelihood_covariance_mode", "full") if cfg is not None else "full"

        self.pca = None

        self.mu_pos = None
        self.mu_neg = None

        # full/shared mode
        self.sigma_pos_inv = None
        self.sigma_neg_inv = None
        self.log_det_pos = None
        self.log_det_neg = None

        self.shared_sigma_inv = None
        self.shared_log_det = None

        # diagonal mode
        self.var_pos = None
        self.var_neg = None
        self.shared_var = None

        # Prior correction & adaptive threshold
        self.log_prior_ratio = 0.0
        self.tau = 0.0  # threshold calibrated on training data

    def _transform(self, features: np.ndarray) -> np.ndarray:
        X = features.astype(np.float64)

        if self.use_standardization:
            X = (X - self.feat_mean) / self.feat_std

        if self.pca is not None:
            X = self.pca.transform(X)

        return X

    def _auto_select_covariance_mode(self, n_pos: int, n_neg: int, d: int) -> str:
        requested = self.covariance_mode.lower()
        min_n = min(n_pos, n_neg)

        if requested == "full" and min_n < 2 * d:
            logger.warning(
                f"Too few samples (min_n={min_n}, d={d}) for full covariance. "
                f"Falling back to 'diagonal'."
            )
            return "diagonal"
        if requested == "shared" and min_n < 2 * d:
            logger.warning(
                f"Too few samples for shared covariance. Falling back to 'diagonal'."
            )
            return "diagonal"
        return requested

    def fit(self, features_pos: np.ndarray, features_neg: np.ndarray):
        features_pos = np.nan_to_num(features_pos, nan=0.0, posinf=0.0, neginf=0.0)
        features_neg = np.nan_to_num(features_neg, nan=0.0, posinf=0.0, neginf=0.0)
        X_all = np.vstack([features_pos, features_neg]).astype(np.float64)

        self.feat_mean = X_all.mean(axis=0)
        self.feat_std = _safe_std(X_all, axis=0)

        Xp = features_pos.astype(np.float64)
        Xn = features_neg.astype(np.float64)

        if self.use_standardization:
            Xp = (Xp - self.feat_mean) / self.feat_std
            Xn = (Xn - self.feat_mean) / self.feat_std

        if self.pca_dim > 0:
            max_dim = min(self.pca_dim, Xp.shape[1], Xn.shape[1], len(X_all) - 1)
            if max_dim >= 1:
                self.pca = PCA(n_components=max_dim, random_state=42)
                self.pca.fit(np.vstack([Xp, Xn]))
                Xp = self.pca.transform(Xp)
                Xn = self.pca.transform(Xn)

        self.mu_pos = Xp.mean(axis=0)
        self.mu_neg = Xn.mean(axis=0)

        # Log prior ratio correction for unequal pseudo-label sizes
        n_pos = len(Xp)
        n_neg = len(Xn)
        if n_pos > 0 and n_neg > 0:
            self.log_prior_ratio = float(np.log(n_pos / n_neg))
        else:
            self.log_prior_ratio = 0.0

        d = Xp.shape[1]
        mode = self._auto_select_covariance_mode(n_pos, n_neg, d)

        if mode == "full":
            sigma_pos = LedoitWolf().fit(Xp).covariance_
            sigma_neg = LedoitWolf().fit(Xn).covariance_

            self.sigma_pos_inv = np.linalg.inv(sigma_pos)
            self.sigma_neg_inv = np.linalg.inv(sigma_neg)
            self.log_det_pos = np.linalg.slogdet(sigma_pos)[1]
            self.log_det_neg = np.linalg.slogdet(sigma_neg)[1]

        elif mode == "shared":
            sigma_pos = LedoitWolf().fit(Xp).covariance_
            sigma_neg = LedoitWolf().fit(Xn).covariance_

            n_pos = len(Xp)
            n_neg = len(Xn)
            shared_sigma = ((n_pos - 1) * sigma_pos + (n_neg - 1) * sigma_neg) / max(n_pos + n_neg - 2, 1)

            self.shared_sigma_inv = np.linalg.inv(shared_sigma)
            self.shared_log_det = np.linalg.slogdet(shared_sigma)[1]

        elif mode == "diagonal":
            self.var_pos = np.var(Xp, axis=0) + 1e-8
            self.var_neg = np.var(Xn, axis=0) + 1e-8

        else:
            raise ValueError(
                f"Unknown likelihood_covariance_mode={self.covariance_mode}. "
                f"Choose from ['full', 'shared', 'diagonal']"
            )

        self._actual_mode = mode

        # Calibrate threshold on training data using Youden's J
        train_scores_pos = np.array([self._log_likelihood_ratio(x) for x in Xp])
        train_scores_neg = np.array([self._log_likelihood_ratio(x) for x in Xn])
        all_train_scores = np.concatenate([train_scores_pos, train_scores_neg])
        thresholds = np.sort(np.unique(all_train_scores))

        best_tau = 0.0
        best_j = -1.0
        for t in thresholds:
            tpr = np.mean(train_scores_pos >= t)
            fpr = np.mean(train_scores_neg >= t)
            j = tpr + (1.0 - fpr) - 1.0
            if j > best_j:
                best_j = j
                best_tau = t

        self.tau = float(best_tau)

        logger.info(
            f"Likelihood ratio attack fitted "
            f"(pca_dim={self.pca_dim}, covariance_mode={self.covariance_mode}, "
            f"actual_mode={mode}, tau={self.tau:.4f}, J={best_j:.4f}, "
            f"log_prior_ratio={self.log_prior_ratio:.4f})"
        )

    def _log_likelihood_ratio(self, x: np.ndarray) -> float:
        mode = self._actual_mode if hasattr(self, "_actual_mode") else self.covariance_mode.lower()

        if mode == "full":
            ll_pos = _logpdf_full_gaussian(x, self.mu_pos, self.sigma_pos_inv, self.log_det_pos)
            ll_neg = _logpdf_full_gaussian(x, self.mu_neg, self.sigma_neg_inv, self.log_det_neg)
            return ll_pos - ll_neg + self.log_prior_ratio

        if mode == "shared":
            ll_pos = _logpdf_full_gaussian(x, self.mu_pos, self.shared_sigma_inv, self.shared_log_det)
            ll_neg = _logpdf_full_gaussian(x, self.mu_neg, self.shared_sigma_inv, self.shared_log_det)
            return ll_pos - ll_neg + self.log_prior_ratio

        if mode == "diagonal":
            ll_pos = _logpdf_diag_gaussian(x, self.mu_pos, self.var_pos)
            ll_neg = _logpdf_diag_gaussian(x, self.mu_neg, self.var_neg)
            return ll_pos - ll_neg + self.log_prior_ratio

        raise ValueError(f"Unknown covariance mode: {self.covariance_mode}")

    def score(self, features: np.ndarray) -> np.ndarray:
        X = self._transform(features)
        return np.array([self._log_likelihood_ratio(x) for x in X], dtype=np.float64)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.score(features) >= self.tau).astype(np.int32)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClassifierAttack:
    def __init__(self, cfg: AttackConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model = None
        self.feat_mean = None
        self.feat_std = None

    def fit(self, features_pos: np.ndarray, features_neg: np.ndarray):
        features_pos = np.nan_to_num(features_pos, nan=0.0, posinf=0.0, neginf=0.0)
        features_neg = np.nan_to_num(features_neg, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.vstack([features_pos, features_neg]).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.concatenate([
            np.ones(len(features_pos), dtype=np.float32),
            np.zeros(len(features_neg), dtype=np.float32),
        ])

        self.feat_mean = X.mean(axis=0)
        self.feat_std = X.std(axis=0) + 1e-8
        X = (X - self.feat_mean) / self.feat_std

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.cfg.mlp_batch_size, shuffle=True)

        self.model = MLPClassifier(
            input_dim=X.shape[1],
            hidden_dims=self.cfg.mlp_hidden_dims,
            dropout=getattr(self.cfg, "mlp_dropout", 0.2),
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.mlp_lr,
            weight_decay=self.cfg.mlp_weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(self.cfg.mlp_epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.model(xb)
                loss = criterion(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(xb)

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataset)
                logger.info(
                    f"  MLP epoch {epoch + 1}/{self.cfg.mlp_epochs}, loss={avg_loss:.4f}"
                )

        logger.info("Classifier attack fitted")

    def score(self, features: np.ndarray) -> np.ndarray:
        X = features.astype(np.float32)
        X = (X - self.feat_mean) / self.feat_std
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_t)).cpu().numpy()
        return probs

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.score(features) >= 0.5).astype(np.int32)


ATTACK_MAP = {
    "threshold": ThresholdAttack,
    "likelihood": LikelihoodRatioAttack,
    "classifier": ClassifierAttack,
}


def create_attack(cfg: AttackConfig, device: str = "cuda"):
    cfg = copy.deepcopy(cfg)

    if cfg.strategy == "classifier":
        return ClassifierAttack(cfg, device)

    if cfg.strategy == "threshold":
        return ThresholdAttack(cfg)

    if cfg.strategy == "likelihood":
        return LikelihoodRatioAttack(cfg)

    raise ValueError(f"Unknown strategy: {cfg.strategy}")