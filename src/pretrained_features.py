"""
pretrained_features.py
=======================
Transfer learning: use a ResNet18 that was already trained on ImageNet
(1.28M photos, 1000 categories) as a fixed, frozen feature extractor,
instead of training a CNN from scratch (that's what autoencoder.py does).

Why this works at all for bottles, which aren't an ImageNet category: the
early and middle layers of a CNN trained on a huge, diverse photo dataset
learn general-purpose visual features — edges, textures, color blobs,
simple shapes — that turn out to be useful for almost any image, not just
the specific 1000 classes it was trained to name. We reuse those learned
features and skip the fc classification head entirely (we don't want
ImageNet's 1000-class predictions, just the 512-number feature vector that
comes right before them).

We DO NOT fine-tune (update) the backbone's weights here — it stays frozen
(requires_grad=False, model.eval()). This keeps Stage 3's two models
genuinely different techniques worth comparing: fully-learned-from-scratch
(autoencoder) vs. reused-from-elsewhere (this module).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def build_resnet_feature_extractor(device: torch.device):
    """
    Load a ResNet18, strip its final classification layer, and freeze it.

    Tries to load real ImageNet-pretrained weights first. If that download
    fails (e.g. no internet access to download.pytorch.org, or a firewalled
    environment), falls back to a randomly-initialized ResNet18 with an
    explicit warning — this is NOT transfer learning anymore if that
    happens, and any results from it must be treated as a pipeline
    smoke-test only, not a real comparison.

    Returns
    -------
    (model, weights_are_pretrained: bool, preprocess: callable)
    """
    from torchvision.models import resnet18, ResNet18_Weights

    weights_are_pretrained = True
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        preprocess = weights.transforms()  # exact preprocessing recipe the weights were trained with
    except Exception as e:
        print(f"WARNING: could not download pretrained ImageNet weights ({type(e).__name__}: {e}).")
        print("Falling back to a RANDOMLY INITIALIZED ResNet18 so the pipeline can still run.")
        print("This is no longer transfer learning — treat any results as a code smoke-test only.")
        weights_are_pretrained = False
        model = resnet18(weights=None)
        # Match the standard ImageNet preprocessing recipe even for the
        # random-weights fallback, so the rest of the pipeline is identical.
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    model.fc = nn.Identity()  # remove the 1000-class classification head; keep the 512-dim pooled features
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    return model, weights_are_pretrained, preprocess


def build_vit_feature_extractor(device: torch.device):
    """
    Load a ViT-B/16, strip its classification head, and freeze it — the
    Vision Transformer counterpart to build_resnet_feature_extractor.

    Why we don't train a ViT from scratch here: Vision Transformers have no
    built-in assumptions about images (a CNN's convolution already assumes
    "nearby pixels are related" and "patterns look the same everywhere in
    the image" — a ViT has to learn both of those from data instead of
    getting them for free). That's why every practical ViT is pretrained on
    huge datasets first — with our ~177 training images, a from-scratch ViT
    would perform badly and wouldn't tell us anything except "ViTs need a
    lot of data," which is itself one of the strengths/weaknesses this
    stage is meant to surface, not something worth re-demonstrating badly.

    Same pretrained-weights-with-fallback behavior as the ResNet extractors.
    """
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    weights_are_pretrained = True
    try:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        preprocess = weights.transforms()
    except Exception as e:
        print(f"WARNING: could not download pretrained ImageNet weights ({type(e).__name__}: {e}).")
        print("Falling back to a RANDOMLY INITIALIZED ViT-B/16 so the pipeline can still run.")
        print("This is no longer transfer learning — treat any results as a code smoke-test only.")
        weights_are_pretrained = False
        model = vit_b_16(weights=None)
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    model.heads = nn.Identity()  # remove the 1000-class head; keep the 768-dim CLS token embedding
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)

    return model, weights_are_pretrained, preprocess


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run every batch in `loader` through the frozen backbone, return stacked (n, 512) features."""
    all_features = []
    for images, _labels in loader:
        images = images.to(device)
        feats = model(images)
        all_features.append(feats.cpu().numpy())
    return np.vstack(all_features)


def build_resnet_spatial_extractor(device: torch.device):
    """
    Build a truncated ResNet18 that stops after layer2, producing a spatial
    feature map instead of one pooled vector per image — needed for
    per-location (patch-level) anomaly localization in Stage 4, rather than
    the single whole-image embedding used in Stage 3.

    For a 224x224 input, this outputs (128, 28, 28): a 28x28 grid of
    28x28-receptive-field "patches", each described by a 128-number
    feature vector. This is the same idea behind PaDiM, a well-known
    MVTec AD localization technique — we simplify PaDiM's per-location
    Gaussian + Mahalanobis distance down to a per-location mean + Euclidean
    distance, to keep the method tractable to implement and explain here.

    Same pretrained-weights-with-fallback behavior as
    build_resnet_feature_extractor.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    weights_are_pretrained = True
    try:
        weights = ResNet18_Weights.IMAGENET1K_V1
        full_model = resnet18(weights=weights)
        preprocess = weights.transforms()
    except Exception as e:
        print(f"WARNING: could not download pretrained ImageNet weights ({type(e).__name__}: {e}).")
        print("Falling back to a RANDOMLY INITIALIZED ResNet18 so the pipeline can still run.")
        weights_are_pretrained = False
        full_model = resnet18(weights=None)
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    spatial_model = nn.Sequential(*list(full_model.children())[:6])  # conv1..layer2
    spatial_model.eval()
    for param in spatial_model.parameters():
        param.requires_grad = False
    spatial_model = spatial_model.to(device)

    return spatial_model, weights_are_pretrained, preprocess


@torch.no_grad()
def extract_spatial_features(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Run every batch through the truncated backbone, return stacked (n, C, H, W) spatial feature maps."""
    all_features = []
    for images, _labels in loader:
        images = images.to(device)
        feats = model(images)
        all_features.append(feats.cpu().numpy())
    return np.vstack(all_features)
