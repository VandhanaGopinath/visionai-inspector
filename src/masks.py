"""
masks.py
========
Utilities for MVTec AD's pixel-level ground_truth masks and for evaluating
a predicted anomaly heatmap against them.

MVTec's convention: every defective test image has a matching binary mask
under ground_truth/<defect_type>/<same_filename>_mask.png. Normal ("good")
test images have no mask at all — they simply have no defect, so the
"ground truth" is implicitly an all-zero mask the same size as the image.
"""

from pathlib import Path
import numpy as np
from PIL import Image


def find_mask_path(image_filepath: str) -> Path | None:
    """
    Given a test image path like .../test/broken_large/000.png, return the
    matching mask path .../ground_truth/broken_large/000_mask.png.

    Returns None for images under test/good/ (MVTec provides no mask there
    — there's nothing to localize).
    """
    path = Path(image_filepath)
    if path.parent.name == "good":
        return None
    mask_path = path.parent.parent.parent / "ground_truth" / path.parent.name / f"{path.stem}_mask{path.suffix}"
    return mask_path


def load_mask_array(image_filepath: str, target_size: tuple[int, int]) -> np.ndarray:
    """
    Load (or synthesize, for 'good' images) a binary mask for an image,
    resized to target_size (width, height) to match whatever resolution
    the anomaly heatmap being compared against uses.

    Returns
    -------
    np.ndarray of shape (target_size[1], target_size[0]), dtype uint8,
    values in {0, 1} — 1 marks a defective pixel.
    """
    mask_path = find_mask_path(image_filepath)
    if mask_path is None or not mask_path.exists():
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    with Image.open(mask_path) as m:
        m = m.convert("L").resize(target_size, Image.NEAREST)
        arr = (np.array(m) > 127).astype(np.uint8)
    return arr


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection-over-Union between two binary masks. Returns NaN if both are empty (undefined)."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return float("nan")
    return intersection / union


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Dice coefficient (F1 over pixels) between two binary masks. Returns NaN if both are empty."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    denom = pred_mask.sum() + true_mask.sum()
    if denom == 0:
        return float("nan")
    return 2 * intersection / denom


def bounding_box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Return (x_min, y_min, x_max, y_max) around the largest connected
    anomalous region in a binary mask, or None if the mask is empty.

    Uses scipy.ndimage for connected-component labeling — a classical,
    non-deep-learning way to turn a pixel mask into a bounding box, which
    is what a downstream 'object detection' view of this problem would
    actually need.
    """
    from scipy import ndimage

    if mask.sum() == 0:
        return None

    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return None

    component_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    largest_component = np.argmax(component_sizes) + 1
    ys, xs = np.where(labeled == largest_component)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
