"""
inference.py
=============
Single-image inference wrappers for the Streamlit app. This module does
NOT implement new detection logic — it wraps the exact same model
architectures, feature extraction, and thresholding approach already built
and validated in Stages 3-4 (src/pretrained_features.py,
src/autoencoder.py, src/evaluation.py), applied to one uploaded image
instead of a batch from a CSV split.

Important, honest scope note: the underlying CV pipeline is an ANOMALY
DETECTOR (Normal vs. Defective), the same as Stages 2-5. It was never
trained to classify *which* defect type an image shows — Stages 2-5 only
ever measured Normal-vs-Defective separation. This module does not invent
a defect-type classification it doesn't have.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from pretrained_features import build_resnet_feature_extractor
from autoencoder import ConvAutoencoder

AE_EVAL_SIZE = (64, 64)


def load_anomaly_model(models_dir: Path, device: torch.device):
    """
    Load the exported ResNet18 + OneClassSVM anomaly detector.

    Returns a dict: {resnet, preprocess, scaler, ocsvm, threshold,
    resnet_pretrained}. Raises FileNotFoundError with a clear message if
    the export script hasn't been run yet.
    """
    bundle_path = models_dir / "resnet_ocsvm_bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"{bundle_path} not found. Run `python scripts/export_inference_models.py` first "
            "to fit and save the anomaly detector (this app does not train models itself)."
        )
    bundle = joblib.load(bundle_path)

    resnet, resnet_pretrained, preprocess = build_resnet_feature_extractor(device)

    return {
        "resnet": resnet,
        "preprocess": preprocess,
        "scaler": bundle["scaler"],
        "ocsvm": bundle["ocsvm"],
        "threshold": bundle["threshold"],
        "resnet_pretrained": resnet_pretrained,
    }


@torch.no_grad()
def predict_anomaly(image: Image.Image, model_bundle: dict, device: torch.device) -> dict:
    """
    Run the loaded ResNet18+OCSVM anomaly detector on a single PIL image.

    Returns
    -------
    dict with keys: label ("Normal"/"Defective"), score (raw OCSVM
    decision_function value), threshold, confidence (0-1 heuristic, NOT a
    calibrated probability -- see note below).
    """
    tensor = model_bundle["preprocess"](image.convert("RGB")).unsqueeze(0).to(device)
    features = model_bundle["resnet"](tensor).cpu().numpy()
    features_scaled = model_bundle["scaler"].transform(features)
    score = float(model_bundle["ocsvm"].decision_function(features_scaled)[0])
    threshold = model_bundle["threshold"]

    label = "Normal" if score >= threshold else "Defective"

    # Confidence is a simple monotonic squashing of distance-from-threshold,
    # NOT a calibrated probability -- OCSVM's decision_function has no
    # natural probabilistic interpretation. It's shown as a relative
    # indicator of "how far past the decision boundary", not P(defective).
    margin = score - threshold
    confidence = float(1 / (1 + np.exp(-margin)))
    if label == "Defective":
        confidence = 1 - confidence

    return {"label": label, "score": score, "threshold": threshold, "confidence": confidence}


def load_autoencoder_model(weights_path: Path, device: torch.device) -> ConvAutoencoder:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Run `python scripts/export_inference_models.py` first, "
            "or run notebooks/03_deep_learning_anomaly_detection.ipynb to produce it."
        )
    model = ConvAutoencoder(latent_dim=64, dropout_p=0.1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def compute_localization(image: Image.Image, autoencoder: ConvAutoencoder, pixel_threshold: float,
                          device: torch.device, eval_size: tuple[int, int] = AE_EVAL_SIZE) -> dict:
    """
    Reuses Stage 4's exact approach (per-pixel autoencoder reconstruction
    error) to localize likely anomalous regions in a single image.

    Returns
    -------
    dict with keys: resized_image (np.ndarray, eval_size), heatmap
    (np.ndarray, eval_size), predicted_mask (binary np.ndarray),
    bounding_box (x0,y0,x1,y1 or None).
    """
    from torchvision import transforms as T

    transform = T.Compose([T.Resize(eval_size), T.ToTensor()])
    resized_image = np.array(image.convert("RGB").resize(eval_size))

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    reconstruction = autoencoder(tensor)
    error_map = ((reconstruction - tensor) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

    predicted_mask = (error_map > pixel_threshold).astype(np.uint8)

    bounding_box = None
    if predicted_mask.sum() > 0:
        labeled, n_components = ndimage.label(predicted_mask)
        if n_components > 0:
            sizes = ndimage.sum(predicted_mask, labeled, range(1, n_components + 1))
            largest = np.argmax(sizes) + 1
            ys, xs = np.where(labeled == largest)
            bounding_box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    return {
        "resized_image": resized_image,
        "heatmap": error_map,
        "predicted_mask": predicted_mask,
        "bounding_box": bounding_box,
    }
