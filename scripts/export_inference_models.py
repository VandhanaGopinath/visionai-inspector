"""
export_inference_models.py
============================
Fits and saves the artifacts the Streamlit app needs to run inference
without retraining anything itself.

This script does NOT implement new detection logic. It calls the exact
same functions already built and validated in Stages 1/3/4:
- data_utils.build_protocol_splits          (Stage 1's protocol-preserving split)
- pretrained_features.build_resnet_feature_extractor / extract_features  (Stage 3, Model B)
- sklearn StandardScaler + OneClassSVM with the same hyperparameters used in Stage 3
- evaluation.calibrate_threshold             (Stage 3's val-only threshold calibration)
- autoencoder.ConvAutoencoder                (Stage 3, Model A -- loads Stage 3's saved weights if present)
- evaluation.calibrate_pixel_threshold       (Stage 4's per-pixel threshold calibration)

Why this script exists at all: the notebooks fit these models fresh every
time they run (fine for exploration/evaluation), but never saved the
fitted scaler/OCSVM/threshold to disk. A live app needs a persisted,
already-trained model to load instantly, not a notebook to re-execute on
every request. This script is the one-time (or re-run-when-you-want-to-
retrain) step that produces those saved artifacts.

Usage:
    python scripts/export_inference_models.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from data_utils import build_protocol_splits, make_synthetic_placeholder_dataset
from torch_dataset import BottleDataset
from pretrained_features import build_resnet_feature_extractor, extract_features
from evaluation import calibrate_threshold, calibrate_pixel_threshold
from autoencoder import ConvAutoencoder
import monitoring

CATEGORY = "bottle"
RANDOM_SEED = 42
THRESHOLD_PERCENTILE = 5
AE_EVAL_SIZE = (64, 64)
BATCH_SIZE = 16


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_root = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"
    processed_root = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    category_path = raw_root / CATEGORY
    using_synthetic = not category_path.exists()
    if using_synthetic:
        print("Real MVTec data not found -- generating synthetic placeholder so this script is runnable.")
        print("Exported models from synthetic data are for PIPELINE TESTING ONLY, not real inspection use.")
        make_synthetic_placeholder_dataset(raw_root, category=CATEGORY, seed=RANDOM_SEED)

    train_csv = processed_root / f"{CATEGORY}_train.csv"
    val_csv = processed_root / f"{CATEGORY}_val.csv"
    if train_csv.exists() and val_csv.exists():
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        print("Stage 1 split CSVs not found -- regenerating with the same protocol-preserving logic.")
        train_df, val_df, _ = build_protocol_splits(category_path, val_frac=0.15, seed=RANDOM_SEED)

    print(f"train (normal only): {len(train_df)}   val (normal only): {len(val_df)}")

    # ---------------------------------------------------------------- Model B: ResNet18 + OCSVM
    print("\n--- Fitting ResNet18 + One-Class SVM (same hyperparameters as Stage 3) ---")
    resnet, resnet_pretrained, preprocess = build_resnet_feature_extractor(device)

    train_loader = DataLoader(BottleDataset(train_df, transform=preprocess), batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(BottleDataset(val_df, transform=preprocess), batch_size=BATCH_SIZE, shuffle=False)

    X_train = extract_features(resnet, train_loader, device)
    X_val = extract_features(resnet, val_loader, device)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    ocsvm.fit(X_train_scaled)

    val_scores = ocsvm.decision_function(X_val_scaled)
    threshold = calibrate_threshold(val_scores, THRESHOLD_PERCENTILE, higher_is_anomalous=False)

    bundle = {"scaler": scaler, "ocsvm": ocsvm, "threshold": threshold, "resnet_pretrained": resnet_pretrained}
    joblib.dump(bundle, models_dir / "resnet_ocsvm_bundle.joblib")
    print(f"Saved -> {models_dir / 'resnet_ocsvm_bundle.joblib'}  (threshold={threshold:.4f}, pretrained={resnet_pretrained})")

    # Stage 10 reference: score + image-statistic distributions of NORMAL validation images, plus a
    # version hash of the model file, so the monitoring module can later detect drift in production.
    from PIL import Image
    val_stats = [monitoring.image_stats(Image.open(fp).convert("RGB")) for fp in val_df["filepath"]]
    reference = monitoring.build_reference(
        scores=val_scores,
        brightness=[v["brightness"] for v in val_stats],
        contrast=[v["contrast"] for v in val_stats],
        model_version=monitoring.compute_model_version(models_dir / "resnet_ocsvm_bundle.joblib"),
        source="normal validation images" + (" (SYNTHETIC PLACEHOLDER)" if using_synthetic else ""),
    )
    monitoring.save_reference(reference, models_dir / "monitoring_reference.json")
    print(f"Saved -> {models_dir / 'monitoring_reference.json'}  (model_version={reference['model_version']})")

    # ---------------------------------------------------------------- Model A: autoencoder (reuse Stage 3 weights if present)
    print("\n--- Preparing autoencoder for localization (Stage 4's approach) ---")
    autoencoder = ConvAutoencoder(latent_dim=64, dropout_p=0.1).to(device)
    stage3_weights = processed_root / f"{CATEGORY}_autoencoder_weights.pt"

    if stage3_weights.exists():
        autoencoder.load_state_dict(torch.load(stage3_weights, map_location=device))
        print(f"Loaded existing trained weights from {stage3_weights} (no retraining).")
    else:
        print("No Stage 3 autoencoder weights found -- training a fresh copy (same architecture/settings).")
        ae_transform = transforms.Compose([transforms.Resize(AE_EVAL_SIZE), transforms.ToTensor()])
        ae_train_loader = DataLoader(BottleDataset(train_df, transform=ae_transform), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        autoencoder.train()
        for epoch in range(30):
            for images, _ in ae_train_loader:
                images = images.to(device)
                optimizer.zero_grad()
                loss = loss_fn(autoencoder(images), images)
                loss.backward()
                optimizer.step()

    autoencoder.eval()
    ae_weights_out = models_dir / "autoencoder_weights.pt"
    torch.save(autoencoder.state_dict(), ae_weights_out)
    print(f"Saved -> {ae_weights_out}")

    # Pixel-level threshold, calibrated on normal validation heatmaps only (Stage 4's approach).
    ae_transform = transforms.Compose([transforms.Resize(AE_EVAL_SIZE), transforms.ToTensor()])
    val_loader_ae = DataLoader(BottleDataset(val_df, transform=ae_transform), batch_size=BATCH_SIZE, shuffle=False)

    val_heatmaps = []
    with torch.no_grad():
        for images, _ in val_loader_ae:
            images = images.to(device)
            recon = autoencoder(images)
            error_map = ((recon - images) ** 2).mean(dim=1)
            val_heatmaps.append(error_map.cpu().numpy())
    val_heatmaps = np.concatenate(val_heatmaps, axis=0)

    pixel_threshold = calibrate_pixel_threshold(val_heatmaps, THRESHOLD_PERCENTILE)
    joblib.dump({"pixel_threshold": pixel_threshold, "eval_size": AE_EVAL_SIZE}, models_dir / "autoencoder_threshold.joblib")
    print(f"Saved -> {models_dir / 'autoencoder_threshold.joblib'}  (pixel_threshold={pixel_threshold:.6f})")

    print("\nDone. Run `streamlit run app.py` to launch the app.")
    if using_synthetic:
        print("\nREMINDER: these artifacts were fit on SYNTHETIC PLACEHOLDER DATA -- re-run this script")
        print("after downloading the real MVTec bottle dataset before using the app for anything real.")


if __name__ == "__main__":
    main()
