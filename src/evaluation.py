"""
evaluation.py
=============
Shared evaluation logic for anomaly-detection models, used identically
across the classical (Stage 2) and deep learning (Stage 3) approaches so
every model is judged the same way.

The core discipline, repeated from Stage 1/2 and enforced here in code
rather than just in comments: a decision threshold is calibrated using ONLY
normal validation scores, never test labels. Test labels are used exactly
once, purely to measure performance after the threshold is already fixed.
"""

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


def calibrate_threshold(val_scores: np.ndarray, percentile: float, higher_is_anomalous: bool) -> float:
    """
    Pick an operating threshold from validation scores (normal images only).

    If higher_is_anomalous (e.g. autoencoder reconstruction error): the
    threshold is the (100 - percentile)-th percentile of validation scores
    — e.g. percentile=5 means "the score above which only 5% of genuinely
    normal validation images fall", i.e. a 5% tolerated false-positive rate.

    If not higher_is_anomalous (e.g. OCSVM/IsolationForest decision_function,
    where higher = more normal): the threshold is the raw `percentile`-th
    percentile — e.g. percentile=5 means the same 5% false-positive
    tolerance, just on the other tail of the distribution.
    """
    if higher_is_anomalous:
        return float(np.percentile(val_scores, 100 - percentile))
    else:
        return float(np.percentile(val_scores, percentile))


def evaluate_anomaly_scores(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float,
                             higher_is_anomalous: bool, tag: str = "") -> dict:
    """
    Apply a fixed threshold to continuous anomaly scores, print a
    classification report, and compute threshold-independent ROC-AUC.

    Returns
    -------
    dict with keys: y_pred, auc — for downstream plotting/comparison.
    """
    if higher_is_anomalous:
        y_pred = (scores > threshold).astype(int)
        auc_input_scores = scores
    else:
        y_pred = (scores < threshold).astype(int)
        auc_input_scores = -scores  # roc_auc_score expects higher score = positive (defective) class

    print(f"=== {tag}{name} ===")
    print(classification_report(y_true, y_pred, target_names=["good", "defective"]))
    auc = roc_auc_score(y_true, auc_input_scores)
    print(f"ROC-AUC: {auc:.4f}")

    return {"y_pred": y_pred, "auc": auc}


def calibrate_pixel_threshold(val_heatmaps: np.ndarray, percentile: float) -> float:
    """
    Calibrate a per-pixel anomaly threshold from a stack of validation
    heatmaps (normal images only, shape (n_images, H, W)) — flattens every
    pixel from every normal validation image into one distribution and
    picks the value below which `percentile`% of them fall.

    Same principle as calibrate_threshold, just applied per-pixel instead
    of per-image: we only ever look at pixels we know are genuinely normal
    to decide where the cutoff should be.
    """
    return float(np.percentile(val_heatmaps.ravel(), 100 - percentile))


def pixel_level_report(true_masks: np.ndarray, pred_masks: np.ndarray) -> dict:
    """
    Flatten every pixel across every test image (both good and defective)
    and compute precision/recall/F1 at the pixel level — a stricter test
    than image-level classification, since it requires getting the *shape*
    of the defect right, not just detecting that one exists somewhere.

    Parameters
    ----------
    true_masks, pred_masks : np.ndarray of shape (n_images, H, W), binary.
    """
    y_true_flat = true_masks.ravel()
    y_pred_flat = pred_masks.ravel()
    print(classification_report(y_true_flat, y_pred_flat, target_names=["normal pixel", "defective pixel"], zero_division=0))
    return {"y_true_flat": y_true_flat, "y_pred_flat": y_pred_flat}
