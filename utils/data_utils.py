import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset
from monai import transforms

from typing import Sequence

from transform_timer import transform_timer

__all__ = ["SegmentationDataset",
           "SegmentationPatchDataset",
           "get_augmentation_transform"]


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str,
                 image_key: str = "image",
                 label_key: str = "label",
                 load_meta: bool = False):
        self.image_key = image_key
        self.label_key = label_key
        self.data_dir = data_dir
        self.load_meta = load_meta

        json_path = os.path.join(data_dir, json_file)
        with open(json_path, "r") as fp:
            data = json.load(fp)
            if data_list_key not in data:
                raise ValueError(f"key '{data_list_key}' not in data")

            self.data_list = data[data_list_key]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Keep images in disk memory
        data = self.load_mmap(idx)

        # Load data to RAM
        data[self.image_key] = data[self.image_key].copy()
        data[self.label_key] = data[self.label_key].copy()

        if self.load_meta:
            data["filename"] = self.data_list[idx][self.image_key]

        return data

    def load_mmap(self, idx):
        # Keep images in disk memory
        image_path = os.path.join(self.data_dir, self.data_list[idx][self.image_key])
        label_path = os.path.join(self.data_dir, self.data_list[idx][self.label_key])
        image = np.load(image_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")

        image = image.reshape((1, *image.shape))
        label = label.reshape((1, *label.shape))

        data = {
            self.image_key: image,
            self.label_key: label
        }
        return data


class SegmentationPatchDataset(SegmentationDataset):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str,
                 patch_size: int | Sequence[int],
                 patch_batch_size: int,
                 image_key: str = "image",
                 label_key: str = "label",
                 transform=None):
        super().__init__(data_dir=data_dir,
                         json_file=json_file,
                         data_list_key=data_list_key,
                         image_key=image_key,
                         label_key=label_key)

        self.patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else patch_size
        self.patch_batch_size = patch_batch_size
        self.transform = transform
        self.cropper = transforms.RandCropByLabelClassesd(
            keys=[image_key, label_key],
            label_key=label_key,
            spatial_size=self.patch_size,
            ratios=[1, 1],
            num_classes=2,
            num_samples=self.patch_batch_size,
            warn=False
        )

        self.cropper.set_random_state(0)

    def __getitem__(self, idx):
        # Keep images in disk memory
        data = self.load_mmap(idx)

        patches = self.cropper(data)

        # Load only patches to RAM
        for i, patch in enumerate(patches):
            patches[i] = {
                self.image_key: torch.clone(patch[self.image_key]),
                self.label_key: torch.clone(patch[self.label_key])
            }

        # Augmentation
        if self.transform:
            patches = self.transform(patches)

        return patches


def get_augmentation_transform(args):
    augmentation_transform = transforms.Compose(
        [
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.rand_scale_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.rand_shift_prob),
            transforms.RandGaussianNoised(keys="image", prob=args.rand_noise_prob, mean=0.0, std=0.05),
            transforms.RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.11, 0.20),
                sigma_y=(0.7, 1.2),
                sigma_z=(0.7, 1.2),
                approx='erf',
                prob=args.rand_smooth_prob),
            transforms.RandAdjustContrastd(
                keys="image",
                prob=1.0,
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=False),
            transforms.RandRotated(
                keys=["image", "label"],
                prob=1.0,
                range_x=torch.pi,
                mode=("bilinear", "nearest"),
                keep_size=True,
                padding_mode="zeros",
            ),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    augmentation_transform.set_random_state(0)
    return augmentation_transform
