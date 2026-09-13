"""
features.py
============
Hand-crafted feature extraction for Stage 2 (classical ML baseline).

Why not just feed raw pixels into a model? A single 128x128 RGB image has
49,152 numbers. Classical algorithms like One-Class SVM and Isolation
Forest don't scale well to that many dimensions relative to how few
training images we have (~180), and raw pixel values don't directly encode
anything about *where* an image deviates from normal. So instead we engineer
a small set of numbers per image that are more informative for this specific
problem: is this bottle's surface uniform the way a normal bottle's is, or
does it show a localized irregularity?

This is exactly the kind of feature engineering a CNN in Stage 3 learns to
do automatically from data. Doing it by hand here is slower and less
powerful, but makes explicit what's actually being detected — which is the
whole point of a classical-ML baseline stage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


def extract_grid_features(image_array: np.ndarray, target_size: tuple[int, int] = (128, 128), grid_size: int = 4) -> np.ndarray:
    """
    Convert an image into a small hand-crafted feature vector.

    Approach:
    1. Grayscale — simple channel average. Defects like scratches, cracks,
       and contamination generally show up as intensity/texture changes
       rather than color-specific changes, so color is dropped to keep the
       feature space small and focused.
    2. Resize to a fixed size so every image produces a feature vector of
       the same length regardless of its original resolution.
    3. Split into a grid_size x grid_size grid of cells. Per cell, compute:
       - mean intensity (is this region unusually bright/dark?)
       - std of intensity (is this region unusually smooth or noisy/textured?)
       - mean gradient magnitude (is this region unusually full of edges —
         a proxy for cracks, scratches, or contamination boundaries?)
       Splitting into a grid rather than computing these three numbers once
       for the whole image is what lets the feature vector capture *where*
       something unusual is happening, not just that the image as a whole
       has some average brightness — a defect in one corner shouldn't be
       averaged away by the rest of a normal bottle.
    4. Append two whole-image statistics (overall mean, overall std) as a
       global summary alongside the local grid statistics.

    Returns
    -------
    np.ndarray of shape (grid_size * grid_size * 3 + 2,) — for the default
    grid_size=4 that's 4*4*3 + 2 = 50 features per image.
    """
    gray = image_array.astype(np.float32).mean(axis=2)
    img = Image.fromarray(gray.astype(np.uint8)).resize(target_size, Image.BILINEAR)
    gray = np.array(img, dtype=np.float32)

    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    h, w = gray.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    features = []
    for i in range(grid_size):
        for j in range(grid_size):
            cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            grad_cell = grad_mag[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            features.append(cell.mean())
            features.append(cell.std())
            features.append(grad_cell.mean())

    features.append(gray.mean())
    features.append(gray.std())

    return np.array(features, dtype=np.float32)


def feature_names(grid_size: int = 4) -> list[str]:
    """Human-readable names matching extract_grid_features' output order — useful for inspection/debugging."""
    names = []
    for i in range(grid_size):
        for j in range(grid_size):
            names.append(f"cell_{i}_{j}_mean")
            names.append(f"cell_{i}_{j}_std")
            names.append(f"cell_{i}_{j}_grad_mean")
    names.append("global_mean")
    names.append("global_std")
    return names


def build_feature_matrix(filepaths: list[str], target_size: tuple[int, int] = (128, 128), grid_size: int = 4) -> np.ndarray:
    """
    Apply extract_grid_features to a list of image filepaths.

    Returns
    -------
    np.ndarray of shape (n_images, n_features)
    """
    rows = []
    for fp in filepaths:
        with Image.open(fp) as img:
            arr = np.array(img.convert("RGB"))
        rows.append(extract_grid_features(arr, target_size=target_size, grid_size=grid_size))
    return np.vstack(rows)
