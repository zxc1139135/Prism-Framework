"""
Calibration module for Prism.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold

from attack import create_attack
from config import AttackConfig
from utils import get_logger

logger = get_logger(__name__)

def _safe_median(x: np.ndarray, axis=0) -> np.ndarray:
    return np.median(x, axis=axis)


def _safe_iqr(x: np.ndarray, axis=0) -> np.ndarray:
    q75 = np.quantile(x, 0.75, axis=axis)
    q25 = np.quantile(x, 0.25, axis=axis)
    return q75 - q25


def _robust_standardize(features: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Robust z-score using median / IQR.
    """
    med = _safe_median(features, axis=0)
    iqr = _safe_iqr(features, axis=0)

    scale = np.where(iqr < 1e-8, 1.0, iqr / 1.349)
    z = (features - med) / (scale + 1e-8)

    meta = {
        "median": med,
        "scale": scale,
    }
    return z, meta

DOMAIN_POLARITY_FULL8 = np.array([+1, -1, +1, +1, +1, +1, +1, -1], dtype=np.float64)
DOMAIN_POLARITY_CLASSIC4 = np.array([+1, -1, +1, +1], dtype=np.float64)
DOMAIN_POLARITY_ROBUST4 = np.array([+1, -1, +1, +1], dtype=np.float64)


def _get_domain_polarity(d: int) -> np.ndarray:
    """Return domain-aware polarity vector for feature dimension d."""
    if d == 8:
        return DOMAIN_POLARITY_FULL8.copy()
    if d == 4:
        return DOMAIN_POLARITY_CLASSIC4.copy()
    # Generic fallback: assume all-positive
    return np.ones(d, dtype=np.float64)


def _estimate_dimension_polarity_and_weight(
    z: np.ndarray,
    polarity_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n, d = z.shape
    polarity = np.ones(d, dtype=np.float64)
    weights = np.ones(d, dtype=np.float64)

    details = []

    if polarity_mode == "domain":
        polarity = _get_domain_polarity(d)
        logger.info(f"Using domain-aware polarity: {polarity.tolist()}")
    else:
        for j in range(d):
            hi_thr = np.quantile(z[:, j], 0.80)
            lo_thr = np.quantile(z[:, j], 0.20)

            hi_mask = z[:, j] >= hi_thr
            lo_mask = z[:, j] <= lo_thr

            if hi_mask.sum() < 5 or lo_mask.sum() < 5:
                polarity[j] = 1.0
                details.append(
                    {
                        "dim": int(j),
                        "polarity": 1.0,
                        "weight": 1.0,
                        "effect": 0.0,
                        "note": "fallback_small_group",
                    }
                )
                continue

            hi_center = z[hi_mask].mean(axis=0)
            lo_center = z[lo_mask].mean(axis=0)

            # FIX: only look at dimension j itself for polarity direction
            direction_signal = float(hi_center[j] - lo_center[j])
            polarity[j] = 1.0 if direction_signal >= 0 else -1.0

            details.append(
                {
                    "dim": int(j),
                    "polarity": float(polarity[j]),
                    "direction_signal": float(direction_signal),
                }
            )

    for j in range(d):
        q75 = np.quantile(z[:, j], 0.75)
        q25 = np.quantile(z[:, j], 0.25)
        spread = abs(float(q75 - q25))
        col = z[:, j]
        mu4 = float(np.mean((col - col.mean()) ** 4))
        sigma2 = float(np.var(col)) + 1e-8
        kurtosis = mu4 / (sigma2 ** 2) - 3.0  # excess kurtosis
        weights[j] = max(spread + 0.1 * max(kurtosis, 0.0), 1e-3)

    weights = weights / (weights.sum() + 1e-8)

    meta = {
        "polarity_mode": polarity_mode,
        "dimension_details": details,
    }
    return polarity, weights, meta


def compute_contrastive_scores(
    features: np.ndarray,
    cfg: AttackConfig = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Build a contrastive score from the feature matrix.
    """
    z, z_meta = _robust_standardize(features)

    polarity_mode = "domain"  # safe default
    if cfg is not None:
        polarity_mode = str(getattr(cfg, "polarity_mode", "domain"))

    polarity, weights, pw_meta = _estimate_dimension_polarity_and_weight(
        z, polarity_mode=polarity_mode
    )

    signed = z * polarity.reshape(1, -1)

    scores = signed @ weights

    meta = {
        "standardization": z_meta,
        "polarity": polarity,
        "weights": weights,
        "polarity_weight_meta": pw_meta,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }

    logger.info(
        "Contrastive scores: "
        f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
        f"min={scores.min():.4f}, max={scores.max():.4f}"
    )
    logger.info(
        f"Estimated polarity={polarity.tolist()}, "
        f"weights={np.round(weights, 4).tolist()}"
    )

    return scores, meta


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm


def _pairwise_center_distance(
    x: np.ndarray,
    center: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    center = center.reshape(1, -1)

    if metric == "cosine":
        xn = _normalize_rows(x)
        cn = _normalize_rows(center)
        return 1.0 - (xn @ cn.T).reshape(-1)

    return np.linalg.norm(x - center, axis=1)


def _top_by_score(idx: np.ndarray, scores: np.ndarray, k: int, largest: bool) -> np.ndarray:
    if len(idx) <= k:
        return idx
    local_scores = scores[idx]
    if largest:
        keep = np.argsort(local_scores)[-k:]
    else:
        keep = np.argsort(local_scores)[:k]
    return idx[keep]


def _safe_cfg_get(cfg: AttackConfig, name: str, default):
    return getattr(cfg, name, default)


def _construct_pseudo_labels_compact(
    features: np.ndarray,
    scores: np.ndarray,
    cfg: AttackConfig,
) -> Dict:
    p = float(_safe_cfg_get(cfg, "quantile_p", 0.90))
    p = float(max(0.50, min(0.99, p)))

    min_per_class = int(_safe_cfg_get(cfg, "pseudo_min_per_class", max(20, min(60, len(features) // 25))))
    compactness_keep_ratio = float(_safe_cfg_get(cfg, "compactness_keep_ratio", 0.5))
    compactness_keep_ratio = float(max(0.10, min(1.0, compactness_keep_ratio)))
    distance_metric = str(_safe_cfg_get(cfg, "compactness_distance_metric", "l2"))

    q_high = np.quantile(scores, p)
    q_low = np.quantile(scores, 1 - p)

    pre_pos_idx = np.where(scores >= q_high)[0]
    pre_neg_idx = np.where(scores <= q_low)[0]

    n = len(scores)
    if len(pre_pos_idx) < min_per_class:
        pre_pos_idx = np.argsort(scores)[-min(min_per_class, n // 2):]
    if len(pre_neg_idx) < min_per_class:
        pre_neg_idx = np.argsort(scores)[:min(min_per_class, n // 2)]

    def _filter(idx: np.ndarray, side: str) -> Tuple[np.ndarray, Dict]:
        if len(idx) <= 2:
            return idx, {
                "before": int(len(idx)),
                "after": int(len(idx)),
                "distance_metric": distance_metric,
                "mean_dist_before": 0.0,
                "mean_dist_after": 0.0,
            }

        X = features[idx]
        center = X.mean(axis=0)
        dist = _pairwise_center_distance(X, center, metric=distance_metric)

        keep_n = max(min_per_class, int(round(len(idx) * compactness_keep_ratio)))
        keep_n = min(keep_n, len(idx))

        order = np.argsort(dist)[:keep_n]
        kept_idx = idx[order]

        meta = {
            "before": int(len(idx)),
            "after": int(len(kept_idx)),
            "distance_metric": distance_metric,
            "mean_dist_before": float(np.mean(dist)),
            "mean_dist_after": float(np.mean(dist[order])),
        }

        logger.info(
            f"[compactness] side={side}, before={len(idx)}, after={len(kept_idx)}, "
            f"mean_dist_before={meta['mean_dist_before']:.4f}, "
            f"mean_dist_after={meta['mean_dist_after']:.4f}"
        )
        return kept_idx, meta

    pos_idx, pos_compact_meta = _filter(pre_pos_idx, "pos")
    neg_idx, neg_compact_meta = _filter(pre_neg_idx, "neg")

    logger.info(
        f"Pseudo-labels (compact): "
        f"pre_pos={len(pre_pos_idx)}, pre_neg={len(pre_neg_idx)}, "
        f"final_pos={len(pos_idx)}, final_neg={len(neg_idx)}, "
        f"(q_high={q_high:.4f}, q_low={q_low:.4f}, p={p:.2f})"
    )

    return {
        "mode": "compact",
        "pos_features": features[pos_idx],
        "neg_features": features[neg_idx],
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "q_high": float(q_high),
        "q_low": float(q_low),
        "pre_filter_pos_idx": pre_pos_idx,
        "pre_filter_neg_idx": pre_neg_idx,
        "compact_pos_meta": pos_compact_meta,
        "compact_neg_meta": neg_compact_meta,
    }


def _construct_pseudo_labels_asymmetric(
    features: np.ndarray,
    scores: np.ndarray,
    cfg: AttackConfig,
) -> Dict:
    pos_quantile = float(_safe_cfg_get(cfg, "pseudo_pos_quantile", 0.97))
    neg_quantile = float(_safe_cfg_get(cfg, "pseudo_neg_quantile", 0.93))

    pos_quantile = float(max(0.50, min(0.999, pos_quantile)))
    neg_quantile = float(max(0.50, min(0.999, neg_quantile)))

    min_pos = int(_safe_cfg_get(cfg, "pseudo_min_pos", max(20, min(60, len(features) // 25))))
    min_neg = int(_safe_cfg_get(cfg, "pseudo_min_neg", max(20, min(60, len(features) // 25))))

    q_high = np.quantile(scores, pos_quantile)
    q_low = np.quantile(scores, 1 - neg_quantile)

    pos_idx = np.where(scores >= q_high)[0]
    neg_idx = np.where(scores <= q_low)[0]

    n = len(scores)
    if len(pos_idx) < min_pos:
        pos_idx = np.argsort(scores)[-min(min_pos, n // 2):]
    if len(neg_idx) < min_neg:
        neg_idx = np.argsort(scores)[:min(min_neg, n // 2)]

    logger.info(
        f"Pseudo-labels (asymmetric): "
        f"{len(pos_idx)} positives, {len(neg_idx)} negatives "
        f"(q_high={q_high:.4f}, q_low={q_low:.4f}, "
        f"pos_q={pos_quantile:.3f}, neg_q={neg_quantile:.3f})"
    )

    return {
        "mode": "asymmetric",
        "pos_features": features[pos_idx],
        "neg_features": features[neg_idx],
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "q_high": float(q_high),
        "q_low": float(q_low),
        "pos_quantile": float(pos_quantile),
        "neg_quantile": float(neg_quantile),
    }


def _construct_pseudo_labels_extreme(
    features: np.ndarray,
    scores: np.ndarray,
    cfg: AttackConfig,
) -> Dict:
    pos_ratio = float(_safe_cfg_get(cfg, "pseudo_pos_ratio", 0.03))
    neg_ratio = float(_safe_cfg_get(cfg, "pseudo_neg_ratio", 0.03))
    min_pos = int(_safe_cfg_get(cfg, "pseudo_min_pos", 20))
    min_neg = int(_safe_cfg_get(cfg, "pseudo_min_neg", 20))
    max_pos = int(_safe_cfg_get(cfg, "pseudo_max_pos", 80))
    max_neg = int(_safe_cfg_get(cfg, "pseudo_max_neg", 80))

    pos_ratio = float(max(0.001, min(0.20, pos_ratio)))
    neg_ratio = float(max(0.001, min(0.20, neg_ratio)))

    n = len(scores)

    pos_k = max(min_pos, int(round(n * pos_ratio)))
    neg_k = max(min_neg, int(round(n * neg_ratio)))

    pos_k = min(pos_k, max_pos, n // 2)
    neg_k = min(neg_k, max_neg, n // 2)

    sorted_idx = np.argsort(scores)
    neg_idx = sorted_idx[:neg_k]
    pos_idx = sorted_idx[-pos_k:]

    q_high = float(np.min(scores[pos_idx])) if len(pos_idx) > 0 else float("nan")
    q_low = float(np.max(scores[neg_idx])) if len(neg_idx) > 0 else float("nan")

    logger.info(
        f"Pseudo-labels (extreme): "
        f"{len(pos_idx)} positives, {len(neg_idx)} negatives "
        f"(pos_ratio={pos_ratio:.4f}, neg_ratio={neg_ratio:.4f}, "
        f"q_high={q_high:.4f}, q_low={q_low:.4f}, "
        f"grey_zone={n - len(pos_idx) - len(neg_idx)})"
    )

    return {
        "mode": "extreme",
        "pos_features": features[pos_idx],
        "neg_features": features[neg_idx],
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "q_high": q_high,
        "q_low": q_low,
        "grey_zone_size": int(n - len(pos_idx) - len(neg_idx)),
    }


def _construct_pseudo_labels_basic(
    features: np.ndarray,
    scores: np.ndarray,
    p: float = 0.95,
    min_per_class: int = 30,
    max_per_class: int = 80,
) -> Dict:
    n = len(scores)
    p = float(max(0.50, min(0.99, p)))

    q_high = np.quantile(scores, p)
    q_low = np.quantile(scores, 1 - p)

    pos_idx = np.where(scores >= q_high)[0]
    neg_idx = np.where(scores <= q_low)[0]

    if len(pos_idx) < min_per_class:
        pos_idx = np.argsort(scores)[-min(min_per_class, n // 2):]
    if len(neg_idx) < min_per_class:
        neg_idx = np.argsort(scores)[:min(min_per_class, n // 2)]

    pos_idx = _top_by_score(pos_idx, scores, max_per_class, largest=True)
    neg_idx = _top_by_score(neg_idx, scores, max_per_class, largest=False)

    logger.info(
        f"Pseudo-labels (selftrain-base): "
        f"{len(pos_idx)} positives, {len(neg_idx)} negatives "
        f"(q_high={q_high:.4f}, q_low={q_low:.4f}, p={p:.2f})"
    )

    return {
        "pos_features": features[pos_idx],
        "neg_features": features[neg_idx],
        "pos_idx": pos_idx,
        "neg_idx": neg_idx,
        "q_high": float(q_high),
        "q_low": float(q_low),
    }


def construct_pseudo_labels(
    features: np.ndarray,
    scores: np.ndarray,
    cfg: AttackConfig,
) -> Dict:
    mode = str(_safe_cfg_get(cfg, "pseudo_label_mode", "compact")).lower()

    if mode == "compact":
        return _construct_pseudo_labels_compact(features, scores, cfg)

    if mode == "asymmetric":
        return _construct_pseudo_labels_asymmetric(features, scores, cfg)

    if mode == "extreme":
        return _construct_pseudo_labels_extreme(features, scores, cfg)

    if mode == "selftrain":
        min_per_class = int(_safe_cfg_get(cfg, "pseudo_min_per_class", 30))
        max_per_class = int(_safe_cfg_get(cfg, "pseudo_max_per_class", 80))
        p = float(_safe_cfg_get(cfg, "init_quantile_p", 0.95))
        result = _construct_pseudo_labels_basic(
            features,
            scores,
            p=p,
            min_per_class=min_per_class,
            max_per_class=max_per_class,
        )
        result["mode"] = "selftrain"
        return result

    raise ValueError(
        f"Unknown pseudo_label_mode={mode}. "
        f"Choose from ['compact', 'asymmetric', 'extreme', 'selftrain']"
    )


# ---------------------------------------------------------------------
# Main calibration API
# ---------------------------------------------------------------------
def calibrate(features: np.ndarray, cfg: AttackConfig, device: str = "cuda"):
    scores, score_meta = compute_contrastive_scores(features, cfg=cfg)
    mode = str(_safe_cfg_get(cfg, "pseudo_label_mode", "compact")).lower()

    if mode == "selftrain":
        init_p = float(_safe_cfg_get(cfg, "init_quantile_p", 0.95))
        refine_p = float(_safe_cfg_get(cfg, "refine_quantile_p", 0.97))
        min_per_class = int(_safe_cfg_get(cfg, "pseudo_min_per_class", 30))
        max_per_class = int(_safe_cfg_get(cfg, "pseudo_max_per_class", 80))

        pseudo_init = _construct_pseudo_labels_basic(
            features,
            scores,
            p=init_p,
            min_per_class=min_per_class,
            max_per_class=max_per_class,
        )

        provisional_model = create_attack(cfg, device)
        provisional_model.fit(
            pseudo_init["pos_features"],
            pseudo_init["neg_features"],
        )

        refined_scores = provisional_model.score(features)

        pos_mean = float(np.mean(refined_scores[pseudo_init["pos_idx"]]))
        neg_mean = float(np.mean(refined_scores[pseudo_init["neg_idx"]]))
        flipped = False
        if pos_mean < neg_mean:
            refined_scores = -refined_scores
            flipped = True
            logger.info("Refinement score direction flipped to keep higher=more member-like")

        pseudo_refined = _construct_pseudo_labels_basic(
            features,
            refined_scores,
            p=refine_p,
            min_per_class=min_per_class,
            max_per_class=max_per_class,
        )

        final_model = create_attack(cfg, device)
        final_model.fit(
            pseudo_refined["pos_features"],
            pseudo_refined["neg_features"],
        )

        logger.info(
            f"Self-training refine: "
            f"init_pos={len(pseudo_init['pos_idx'])}, init_neg={len(pseudo_init['neg_idx'])}, "
            f"refined_pos={len(pseudo_refined['pos_idx'])}, refined_neg={len(pseudo_refined['neg_idx'])}"
        )

        return final_model, {
            "mode": "selftrain",
            "contrastive_scores": scores,
            "contrastive_meta": score_meta,
            "refined_scores": refined_scores,
            "direction_flipped": flipped,
            "initial_pseudo_pos_idx": pseudo_init["pos_idx"],
            "initial_pseudo_neg_idx": pseudo_init["neg_idx"],
            "refined_pseudo_pos_idx": pseudo_refined["pos_idx"],
            "refined_pseudo_neg_idx": pseudo_refined["neg_idx"],
            "init_q_high": pseudo_init["q_high"],
            "init_q_low": pseudo_init["q_low"],
            "refine_q_high": pseudo_refined["q_high"],
            "refine_q_low": pseudo_refined["q_low"],
        }

    pseudo = construct_pseudo_labels(features, scores, cfg)

    attack_model = create_attack(cfg, device)
    attack_model.fit(pseudo["pos_features"], pseudo["neg_features"])

    meta = {
        "mode": pseudo.get("mode", mode),
        "contrastive_scores": scores,
        "contrastive_meta": score_meta,
        "pseudo_pos_idx": pseudo["pos_idx"],
        "pseudo_neg_idx": pseudo["neg_idx"],
        "q_high": pseudo["q_high"],
        "q_low": pseudo["q_low"],
    }

    # optional fields per mode
    for key in [
        "pre_filter_pos_idx",
        "pre_filter_neg_idx",
        "compact_pos_meta",
        "compact_neg_meta",
        "pos_quantile",
        "neg_quantile",
        "grey_zone_size",
    ]:
        if key in pseudo:
            meta[key] = pseudo[key]

    return attack_model, meta


def crossfit_calibrate_and_infer(
    features: np.ndarray,
    cfg: AttackConfig,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[Dict, object, Dict]:
    n = len(features)
    num_folds = max(2, min(getattr(cfg, "crossfit_folds", 5), n))
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    predictions = np.zeros(n, dtype=np.int32)
    scores = np.zeros(n, dtype=np.float64)
    fold_meta = []

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(features), start=1):
        fold_features = features[train_idx]
        attack_model, meta = calibrate(fold_features, cfg, device=device)

        fold_scores = attack_model.score(features[test_idx])
        fold_preds = attack_model.predict(features[test_idx])

        scores[test_idx] = fold_scores
        predictions[test_idx] = fold_preds

        entry = {
            "fold": fold_id,
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "mode": meta.get("mode", "unknown"),
        }

        if "pseudo_pos_idx" in meta:
            entry["pseudo_pos"] = int(len(meta["pseudo_pos_idx"]))
        if "pseudo_neg_idx" in meta:
            entry["pseudo_neg"] = int(len(meta["pseudo_neg_idx"]))
        if "q_high" in meta:
            entry["q_high"] = float(meta["q_high"])
        if "q_low" in meta:
            entry["q_low"] = float(meta["q_low"])

        if meta.get("mode") == "compact":
            entry["pre_filter_pos"] = int(len(meta.get("pre_filter_pos_idx", [])))
            entry["pre_filter_neg"] = int(len(meta.get("pre_filter_neg_idx", [])))

        if meta.get("mode") == "asymmetric":
            entry["pos_quantile"] = float(meta.get("pos_quantile", 0.0))
            entry["neg_quantile"] = float(meta.get("neg_quantile", 0.0))

        if meta.get("mode") == "extreme":
            entry["grey_zone_size"] = int(meta.get("grey_zone_size", 0))

        if meta.get("mode") == "selftrain":
            entry["initial_pseudo_pos"] = int(len(meta.get("initial_pseudo_pos_idx", [])))
            entry["initial_pseudo_neg"] = int(len(meta.get("initial_pseudo_neg_idx", [])))
            entry["refined_pseudo_pos"] = int(len(meta.get("refined_pseudo_pos_idx", [])))
            entry["refined_pseudo_neg"] = int(len(meta.get("refined_pseudo_neg_idx", [])))
            entry["init_q_high"] = float(meta.get("init_q_high", 0.0))
            entry["init_q_low"] = float(meta.get("init_q_low", 0.0))
            entry["refine_q_high"] = float(meta.get("refine_q_high", 0.0))
            entry["refine_q_low"] = float(meta.get("refine_q_low", 0.0))
            entry["direction_flipped"] = bool(meta.get("direction_flipped", False))

        fold_meta.append(entry)

        logger.info(
            f"Cross-fit fold {fold_id}/{num_folds}: "
            f"train={len(train_idx)}, test={len(test_idx)}, mode={entry['mode']}"
        )

    final_model, final_meta = calibrate(features, cfg, device=device)

    return (
        {
            "predictions": predictions,
            "scores": scores,
        },
        final_model,
        {
            "mode": "crossfit",
            "num_folds": num_folds,
            "folds": fold_meta,
            "final_fit": final_meta,
        },
    )