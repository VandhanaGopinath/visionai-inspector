"""
torch_dataset.py
=================
A PyTorch Dataset wrapping our existing metadata table (filepath, label,
defect_type) so it can be used with a DataLoader for batched, shuffled
loading during training.

Why a custom Dataset class: PyTorch's DataLoader needs to know how to fetch
one example at a time (via __getitem__) and how many there are (__len__).
Wrapping our own data this way — rather than loading everything into memory
upfront — is the standard pattern, and it's what lets a DataLoader apply
transforms and batch images on the fly, in parallel worker processes, for
datasets far bigger than could fit in memory (not a concern at our ~300
images, but the pattern is the same regardless of dataset size).
"""

from PIL import Image
from torch.utils.data import Dataset


class BottleDataset(Dataset):
    """
    Parameters
    ----------
    dataframe : pd.DataFrame
        Must have 'filepath' and 'label' columns (as produced by
        build_metadata_dataframe / build_protocol_splits).
    transform : torchvision.transforms callable, optional
        Applied to each PIL image before it's returned. Different models
        need different transforms (see notebook) — e.g. the autoencoder
        wants a small 64x64 [0,1] tensor, while the pretrained ResNet wants
        224x224 with ImageNet normalization — so the transform is injected
        rather than hard-coded here.
    """

    def __init__(self, dataframe, transform=None):
        self.filepaths = dataframe["filepath"].tolist()
        self.labels = (dataframe["label"] == "defective").astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
