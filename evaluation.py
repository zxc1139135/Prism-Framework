"""
Evaluation metrics: ROC-AUC, Accuracy, TPR@FPR.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from utils import get_logger

logger = get_logger(__name__)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
    valid = fpr_arr <= fpr_target
    if not valid.any():
        return 0.0
    return float(tpr_arr[valid][-1])


def evaluate(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    fpr_thresholds: List[float] = None,
) -> Dict:
    if fpr_thresholds is None:
        fpr_thresholds = [0.10, 0.01, 0.001, 0.0001]

    labels = np.asarray(labels).astype(np.int32)
    predictions = np.asarray(predictions).astype(np.int32)
    scores = np.asarray(scores).astype(np.float64)

    results = {}
    try:
        results["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        results["roc_auc"] = 0.5

    results["accuracy"] = float(accuracy_score(labels, predictions))

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    results["precision"] = float(prec)
    results["recall"] = float(rec)
    results["f1"] = float(f1)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    results["tp"] = int(tp)
    results["fp"] = int(fp)
    results["tn"] = int(tn)
    results["fn"] = int(fn)

    for fpr_t in fpr_thresholds:
        results[f"tpr@{fpr_t:.4f}fpr"] = tpr_at_fpr(labels, scores, fpr_t)
    return results


def print_results(results: Dict, title: str = "Evaluation Results") -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)
    logger.info(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    logger.info(
        f"  Accuracy  : {results['accuracy']:.4f} ({results['accuracy'] * 100:.1f}%)"
    )
    logger.info(f"  Precision : {results['precision']:.4f}")
    logger.info(f"  Recall    : {results['recall']:.4f}")
    logger.info(f"  F1 Score  : {results['f1']:.4f}")
    logger.info(
        f"  TP={results['tp']}, FP={results['fp']}, TN={results['tn']}, FN={results['fn']}"
    )
    for key, val in results.items():
        if key.startswith("tpr@"):
            fpr_str = key.replace("tpr@", "").replace("fpr", "")
            logger.info(
                f"  TPR@{float(fpr_str) * 100:.2f}%FPR : {val:.4f} ({val * 100:.1f}%)"
            )
    logger.info("=" * 60 + "\n")
