import json
import os
import sys
import torch

import numpy as np
from torch.utils.data import DataLoader

from typing import Sequence


class SegmentationDataLoader:
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
        image_path = os.path.join(self.data_dir, self.data_list[idx][self.image_key])
        label_path = os.path.join(self.data_dir, self.data_list[idx][self.label_key])
        image = np.load(image_path, mmap_mode="r")
        label = np.load(label_path, mmap_mode="r")

        data = {
            self.image_key: image,
            self.label_key: label
        }
        return data


class SegmentationPatchDataLoader(SegmentationDataLoader):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str,
                 patch_size: int | Sequence[int],
                 patch_batch_size: int,
                 image_key: str = "image",
                 label_key: str = "label"):

        super().__init__(data_dir=data_dir,
                         json_file=json_file,
                         data_list_key=data_list_key,
                         image_key=image_key,
                         label_key=label_key)

        self.patch_size = patch_size
        self.patch_batch_size = patch_batch_size

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        image, label = data[self.image_key], data[self.label_key]

        patches = self.get_patches(image, label, num_patches=self.patch_batch_size)

        return patches

    def get_patches(self, image, label, num_patches=4):
        slices = [None] * 3

        patches = []
        for _ in range(num_patches):
            for i, (image_length, patch_length) in enumerate(zip(image.shape, self.patch_size)):
                if image_length >= patch_length:
                    patch_corner = np.random.randint(0, image_length - patch_length + 1)
                else:
                    patch_corner = 0

                slices[i] = slice(patch_corner, patch_corner + patch_length)

            image_patch = image[*slices]
            label_patch = label[*slices]

            patches.append({
                "image": image_patch,
                "label": label_patch
            })

        return patches


if __name__ == '__main__':
    loader = SegmentationDataLoader(
        data_dir="preprocessed",
        json_file="neov.json",
        data_list_key="training",
    )
