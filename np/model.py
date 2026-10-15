import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TORCH_HOME", "D:/torch_cache")

import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

# Standard CIFAR-10 classes
CIFAR10_ALL_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Default 4-class subset: diverse vehicles and animals
DEFAULT_CLASSES = ["airplane", "automobile", "bird", "cat"]


def get_transforms(img_size: int = 96):
    """
    Returns train and validation/inference transforms.
    Data augmentation applied to training:
      - RandomResizedCrop (upscaling to 96x96 for deep CNN features)
      - RandomHorizontalFlip
      - RandomRotation
      - ColorJitter (brightness, contrast, saturation)
      - Tensor conversion & ImageNet normalization
    """
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    return train_transform, val_transform


class CIFARParquetDataset(Dataset):
    """
    High-performance PyTorch Dataset loaded directly from parquet files.
    Allows filtering by selected classes and limiting samples per class.
    Pre-extracts image bytes and labels into memory lists for zero-overhead batching.
    """
    def __init__(self, parquet_path: str, selected_classes: list = None, transform=None, samples_per_class: int = None):
        super().__init__()
        self.transform = transform
        if selected_classes is None:
            selected_classes = DEFAULT_CLASSES
        self.classes = selected_classes
        
        target_orig_indices = [CIFAR10_ALL_CLASSES.index(c) for c in selected_classes]
        self.label_map = {orig: idx for idx, orig in enumerate(target_orig_indices)}

        parquet_file = pq.ParquetFile(parquet_path)
        needed = {orig: samples_per_class for orig in target_orig_indices} if samples_per_class is not None else None
        counts = {orig: 0 for orig in target_orig_indices}

        self.img_bytes = []
        self.targets = []
        self.orig_labels = []

        for rg in range(parquet_file.num_row_groups):
            if needed is not None and all(counts[c] >= needed[c] for c in target_orig_indices):
                break
            tbl = parquet_file.read_row_group(rg)
            labels = tbl["label"].to_pylist()
            bytes_list = tbl["img"].flatten()[0].to_pylist()

            for b, l in zip(bytes_list, labels):
                if l in self.label_map:
                    if needed is None or counts[l] < needed[l]:
                        self.img_bytes.append(b)
                        self.targets.append(self.label_map[l])
                        self.orig_labels.append(CIFAR10_ALL_CLASSES[l])
                        counts[l] += 1

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        raw = self.img_bytes[idx]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def get_raw_pil(self, idx):
        """Returns raw PIL Image and class name (for visualization / test inference)"""
        raw = self.img_bytes[idx]
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        cls_name = self.orig_labels[idx]
        return img, cls_name


def get_dataloaders(
    data_dir: str = "D:/cifar_data",
    selected_classes: list = None,
    samples_per_class_train: int = 500,
    samples_per_class_val: int = 150,
    batch_size: int = 32,
    img_size: int = 96
):
    """
    Creates train and validation DataLoader instances.
    """
    if selected_classes is None:
        selected_classes = DEFAULT_CLASSES

    train_transform, val_transform = get_transforms(img_size=img_size)

    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    train_ds = CIFARParquetDataset(
        train_path,
        selected_classes=selected_classes,
        transform=train_transform,
        samples_per_class=samples_per_class_train
    )

    val_ds = CIFARParquetDataset(
        test_path,
        selected_classes=selected_classes,
        transform=val_transform,
        samples_per_class=samples_per_class_val
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, selected_classes


class TransferClassifier(nn.Module):
    """
    Transfer Learning Classifier using pretrained MobileNetV2 backbone.
    Architecture:
      - MobileNetV2 features (pretrained on ImageNet)
      - Adaptive Average Pooling
      - Custom classification head:
        Dropout(0.3) -> Linear(1280, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.2) -> Linear(256, num_classes)
    """
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained MobileNetV2
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.last_channel  # 1280

        # Freeze feature extractor if requested
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def unfreeze_top_layers(self, num_blocks: int = 3):
        """Unfreezes top convolutional blocks for fine-tuning."""
        layers = list(self.features.children())
        for layer in layers[-num_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
