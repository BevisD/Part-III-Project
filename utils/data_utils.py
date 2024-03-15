import json
import os

import monai.transforms
import numpy as np
from torch.utils.data import Dataset
from monai import transforms

from typing import Sequence

__all__ = ["SegmentationDataset",
           "SegmentationPatchDataset",
           "get_augmentation_transform"]


def mmap_to_normal(arr):
    normal_arr = np.zeros_like(arr)
    normal_arr[...] = arr[...]
    return normal_arr


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str,
                 image_key: str = "image",
                 label_key: str = "label"):
        self.image_key = image_key
        self.label_key = label_key
        self.data_dir = data_dir

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
        image, label = self.load_mmap(idx)

        normal_image = mmap_to_normal(image)
        normal_label = mmap_to_normal(label)

        data = {
            self.image_key: normal_image,
            self.label_key: normal_label
        }
        return data

    def load_mmap(self, idx):
        # Keep images in disk memory
        image_path = os.path.join(self.data_dir, self.data_list[idx][self.image_key])
        label_path = os.path.join(self.data_dir, self.data_list[idx][self.label_key])
        image = np.load(image_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")

        image = image.reshape((1, *image.shape))
        label = label.reshape((1, *label.shape))
        return image, label


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

        self.patch_size = (patch_size,)*3 if isinstance(patch_size, int) else patch_size
        self.patch_batch_size = patch_batch_size
        self.transform = transform

    def __getitem__(self, idx):
        # Keep images in disk memory
        image, label = self.load_mmap(idx)

        patches = self.get_patches(image, label, num_patches=self.patch_batch_size)

        if self.transform:
            patches = self.transform(patches)

        return patches

    def get_patches(self, image, label, num_patches=4):
        slices = [None] * 3

        patches = []
        for _ in range(num_patches):
            padding = [(0, 0)]
            for i, (image_length, patch_length) in enumerate(zip(image.shape[1:], self.patch_size)):
                if image_length >= patch_length:
                    patch_corner = np.random.randint(0, image_length - patch_length + 1)
                else:
                    patch_corner = 0

                slices[i] = slice(patch_corner, patch_corner + patch_length)
                padding.append((0, max(patch_length - image_length, 0)))

            image_patch = mmap_to_normal(image[:, slices[0], slices[1], slices[2]])
            label_patch = mmap_to_normal(label[:, slices[0], slices[1], slices[2]])

            image_patch = np.pad(image_patch, padding)
            label_patch = np.pad(label_patch, padding)

            patches.append({
                self.image_key: image_patch,
                self.label_key: label_patch
            })

        return patches


def get_augmentation_transform(args):
    augmentation_transform = transforms.Compose(
        [
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.rand_flip_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.rand_rot_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=["image", "label"], factors=0.1, prob=args.rand_scale_prob),
            transforms.RandShiftIntensityd(keys=["image", "label"], offsets=0.1, prob=args.rand_shift_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
