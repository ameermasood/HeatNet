"""Dataset helpers for HeatNet training."""

from __future__ import annotations

import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms


class RGBHeatmapDataset(Dataset):
    def __init__(self, image_dir, heatmap_dir, image_ext=".png", heatmap_ext=".pt"):
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.image_ext = image_ext
        self.heatmap_ext = heatmap_ext
        self.basenames = _collect_balanced_basenames(self.image_dir, self.image_ext)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        name = self.basenames[idx]
        image_path = os.path.join(self.image_dir, name + self.image_ext)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        heatmap_path = os.path.join(self.heatmap_dir, name + self.heatmap_ext)
        heatmaps = torch.load(heatmap_path)
        return image, heatmaps


class RGBDHeatmapDataset(Dataset):
    def __init__(
        self,
        image_dir,
        depth_dir,
        heatmap_dir,
        image_ext=".png",
        depth_ext=".png",
        heatmap_ext=".pt",
    ):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.heatmap_dir = heatmap_dir
        self.image_ext = image_ext
        self.depth_ext = depth_ext
        self.heatmap_ext = heatmap_ext
        self.basenames = _collect_balanced_basenames(self.image_dir, self.image_ext)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, idx):
        name = self.basenames[idx]

        image_path = os.path.join(self.image_dir, name + self.image_ext)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        depth_path = os.path.join(self.depth_dir, name + self.depth_ext)
        depth = Image.open(depth_path)
        depth_np = np.array(depth).astype(np.float32)
        depth_np = (depth_np - 500.0) / 1000.0
        depth_np = np.clip(depth_np, 0.0, 1.0)
        depth = torch.from_numpy(depth_np).unsqueeze(0)

        heatmap_path = os.path.join(self.heatmap_dir, name + self.heatmap_ext)
        heatmaps = torch.load(heatmap_path)
        return image, depth, heatmaps


def make_loaders(
    dataset,
    batch_size=16,
    train_ratio=0.8,
    seed=8,
    num_workers=4,
):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def _collect_balanced_basenames(image_dir, image_ext):
    all_names = sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(image_dir)
        if filename.endswith(image_ext)
    )

    groups = defaultdict(list)
    for name in all_names:
        object_id = name.split("_")[0]
        groups[object_id].append(name)

    chosen = []
    random.seed(42)
    for _, names in groups.items():
        chosen.extend(random.sample(names, len(names)))

    return sorted(chosen)
