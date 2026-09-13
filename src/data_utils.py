"""
data_utils.py
=============
Reusable Stage 1 functions for VisionAI Inspector.

Kept separate from the notebook on purpose: in a real project, loading and
preprocessing logic should live in importable, testable modules, while the
notebook stays focused on exploration, analysis, and visualization.

Every function has a single, clear responsibility so it's easy to explain
in an interview: "this function does X, takes Y, returns Z, and here's why."
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

# MVTec AD stores exactly one "good" (defect-free) class name across every
# category, and one subfolder per defect type otherwise. We rely on that
# convention to build labels straight from folder names, with no manual
# label file needed.
GOOD_LABEL = "good"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def build_metadata_dataframe(category_root: Path) -> pd.DataFrame:
    """
    Walk an MVTec-AD-style category folder and build one row per image.

    Expected layout (standard MVTec AD structure):
        category_root/
            train/good/*.png
            test/good/*.png
            test/<defect_type>/*.png
            ground_truth/<defect_type>/*_mask.png   (optional, not indexed here)

    Parameters
    ----------
    category_root : Path
        Path to a single category, e.g. data/raw/mvtec_ad/bottle

    Returns
    -------
    pd.DataFrame with columns:
        filepath   : str, absolute path to the image
        split      : "train" or "test" (the split MVTec ships, NOT our own
                     train/val/test split — we build that separately later)
        label      : "good" or "defective"
        defect_type: "good" or the specific defect subfolder name
    """
    rows = []
    for split_dir in ["train", "test"]:
        split_path = category_root / split_dir
        if not split_path.exists():
            continue
        for defect_dir in sorted(split_path.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            label = "good" if defect_type == GOOD_LABEL else "defective"
            for img_path in sorted(defect_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    rows.append(
                        {
                            "filepath": str(img_path),
                            "split": split_dir,
                            "label": label,
                            "defect_type": defect_type,
                        }
                    )
    return pd.DataFrame(rows)


def discover_categories(raw_root: Path) -> list[str]:
    """Return the list of category folder names found under raw_root."""
    if not raw_root.exists():
        return []
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])


def check_image_validity(filepath: str) -> dict:
    """
    Open an image and verify it isn't corrupt, recording basic properties.

    Returns a dict rather than raising, so this can be applied across an
    entire DataFrame with df.apply() without stopping on the first bad file
    — corrupt/unreadable files are a real thing in scraped or shipped
    industrial datasets and should be measured, not assumed away.
    """
    try:
        with Image.open(filepath) as img:
            img.verify()  # cheap structural check, doesn't decode pixels
        with Image.open(filepath) as img:  # re-open: verify() closes the file
            width, height = img.size
            mode = img.mode
        return {"is_valid": True, "width": width, "height": height, "mode": mode, "error": None}
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return {"is_valid": False, "width": None, "height": None, "mode": None, "error": str(e)}


def load_image_as_array(filepath: str, resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load an image as a NumPy array, optionally resized, in RGB.

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype uint8, values in [0, 255].
    Kept as uint8 here deliberately — normalization is a separate,
    explicit step (see normalize_image) so the two concerns don't blur
    together.
    """
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, Image.BILINEAR)
        return np.array(img)


def normalize_image(img_array: np.ndarray, method: str = "0_1") -> np.ndarray:
    """
    Normalize a uint8 image array.

    method="0_1"      -> scale to [0, 1] float32 by dividing by 255
    method="mean_std" -> zero-mean, unit-variance per channel using
                          ImageNet statistics (standard when later using
                          transfer-learning backbones in Stage 3)

    Why normalization matters: neural network training is sensitive to the
    scale of its inputs — large, inconsistent pixel magnitudes (0-255) slow
    down and destabilize gradient descent. Scaling inputs to a small,
    consistent range makes optimization faster and more stable, and mean/std
    normalization additionally matches the statistics a pretrained backbone
    was originally trained on, which matters for transfer learning.
    """
    arr = img_array.astype(np.float32)
    if method == "0_1":
        return arr / 255.0
    elif method == "mean_std":
        arr = arr / 255.0
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (arr - imagenet_mean) / imagenet_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def build_protocol_splits(category_root: Path, val_frac: float = 0.15, seed: int = 42):
    """
    Rebuild the Stage 1 protocol-preserving split: train/val carved only
    from MVTec's original normal-only train/good pool, test left completely
    untouched. See notebooks/01_eda_stage1.ipynb section 11 for the full
    reasoning — this function exists so later stages can regenerate the
    exact same split without duplicating that logic or re-deriving it
    differently by accident.

    Returns
    -------
    (train_df, val_df, test_df) — same shape/columns as build_metadata_dataframe.
    """
    from sklearn.model_selection import train_test_split

    df = build_metadata_dataframe(category_root)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    mvtec_train_pool = df[df["split"] == "train"].reset_index(drop=True)
    assert (mvtec_train_pool["defect_type"] == "good").all(), (
        "Expected MVTec train/ to contain only 'good' images."
    )

    train_df, val_df = train_test_split(mvtec_train_pool, test_size=val_frac, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    return train_df, val_df, test_df


def make_synthetic_placeholder_dataset(root: Path, category: str = "bottle", seed: int = 42) -> None:
    """
    Generate a tiny synthetic dataset with MVTec-AD-style folder structure,
    used ONLY to verify the Stage 1+ pipeline runs end-to-end before the real
    dataset has been downloaded.

    IMPORTANT: This produces random noise images, not real defect photos.
    Every statistic/plot computed from this data must be treated as a code
    smoke-test, never as a real result. The notebooks label every such
    output "[SYNTHETIC PLACEHOLDER DATA]" for this reason.

    Also generates matching ground_truth masks for defective test images
    (a synthetic rectangular blob), so Stage 4's localization pipeline has
    something to run against too.
    """
    rng = np.random.default_rng(seed)
    defect_types = ["broken_large", "broken_small", "contamination"]
    category_root = root / category
    counts = {
        ("train", "good"): 12,
        ("test", "good"): 6,
        ("test", "broken_large"): 4,
        ("test", "broken_small"): 3,
        ("test", "contamination"): 3,
    }
    img_size = 64
    for (split, subfolder), n in counts.items():
        out_dir = category_root / split / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            # Slightly different color statistics per class so plots aren't
            # completely meaningless as a *pipeline* test, while still being
            # obviously synthetic (small, uniform-ish noise, no real texture).
            base = 180 if subfolder == "good" else 120
            noise = rng.integers(base - 20, base + 20, size=(img_size, img_size, 3), dtype=np.uint8)

            if subfolder in defect_types:
                # Stamp a synthetic rectangular "defect" blob so the image
                # and its mask are at least consistent with each other.
                x0, y0 = rng.integers(10, 30, size=2)
                x1, y1 = x0 + rng.integers(10, 20), y0 + rng.integers(10, 20)
                noise[y0:y1, x0:x1] = rng.integers(0, 40, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)

                mask = np.zeros((img_size, img_size), dtype=np.uint8)
                mask[y0:y1, x0:x1] = 255
                mask_dir = category_root / "ground_truth" / subfolder
                mask_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(mask_dir / f"{i:03d}_mask.png")

            Image.fromarray(noise).save(out_dir / f"{i:03d}.png")
