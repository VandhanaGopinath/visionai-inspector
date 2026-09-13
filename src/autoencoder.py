"""
autoencoder.py
==============
A small convolutional autoencoder for reconstruction-based anomaly
detection, trained from scratch (no pretrained weights) on normal-only
images.

Core idea: force the network to compress each image down to a small
"latent" vector and then reconstruct the full image from just that vector.
Trained only on normal images, it becomes good at compressing/reconstructing
*normal* patterns specifically. At test time, an image the network has never
seen anything like (a defect) tends to reconstruct poorly — the compression
it learned doesn't have a good way to represent it. Reconstruction error
(how different the output is from the input) becomes the anomaly score.

Every layer type below maps to a specific concept from the project's
Stage 3 curriculum:
- Conv2d          -> convolution / filters (learns local visual patterns)
- MaxPool2d        -> pooling (downsamples, keeps the strongest response
                      in each region, and gives the network some
                      translation tolerance)
- ReLU             -> activation function (introduces non-linearity;
                      without it, stacking layers would collapse to one
                      big linear function no matter how many layers deep)
- Dropout2d        -> regularization against overfitting (randomly zeroes
                      some feature maps during training, so the network
                      can't over-rely on any single one)
- Linear bottleneck -> the actual "compression" — forces all the spatial
                       information through a small fixed-size vector
- Upsample + Conv2d -> the decoder's way of undoing pooling and rebuilding
                       spatial resolution back up to the original image size
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, dropout_p: float = 0.1):
        super().__init__()

        # Encoder: 64x64x3 -> 32x32x16 -> 16x16x32 -> 8x8x64 -> latent_dim
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 32x32x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 16x16x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 8x8x64
            nn.Dropout2d(dropout_p),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU(),
        )

        # Decoder: mirror image of the encoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),  # -> 16x16x32
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),  # -> 32x32x16
            nn.Upsample(scale_factor=2), nn.Conv2d(16, 3, kernel_size=3, padding=1), nn.Sigmoid(),  # -> 64x64x3, in [0,1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        return self.encoder_fc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 8, 8)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
