import json
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms

from .affine import RandAffineTransformd

__all__ = ["SegmentationDataset",
           "SegmentationPatchDataset",
           "MultiTaskDataset",
           "get_intensity_aug",
           "get_affine_aug"]


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str | Sequence[str],
                 image_key: str = "image",
                 label_key: str = "label",
                 transform=None,
                 load_meta: bool = False):
        self.image_key = image_key
        self.label_key = label_key
        self.data_dir = data_dir
        self.transform = transform
        self.load_meta = load_meta

        self.data_list = []
        json_path = os.path.join(data_dir, json_file)
        with open(json_path, "r") as fp:
            data = json.load(fp)

        if isinstance(data_list_key, (list, tuple)):
            for key in data_list_key:
                if key not in data:
                    raise ValueError(f"key '{key}' not in data")
                self.data_list += data[key]
        else:
            self.data_list = data[data_list_key]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Don't use mmap as loading whole image
        data = self.load_data(idx, mmap=False)

        # Convert from numpy.ndarray to torch.Tensor
        data[self.image_key] = torch.as_tensor(data[self.image_key])
        data[self.label_key] = torch.as_tensor(data[self.label_key])

        if self.transform is not None:
            data = self.transform(data)

        if self.load_meta:
            data["filename"] = self.data_list[idx][self.label_key]

        return data

    def load_data(self, idx: int, mmap: bool = True):
        # Keep images in disk memory
        image_path = os.path.join(self.data_dir, self.data_list[idx][self.image_key])
        label_path = os.path.join(self.data_dir, self.data_list[idx][self.label_key])

        image = np.load(image_path, mmap_mode="r" if mmap else None)
        label = np.load(label_path, mmap_mode="r" if mmap else None)

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
                 data_list_key: str | Sequence[str],
                 patch_size: int | Sequence[int],
                 patch_batch_size: int,
                 image_key: str = "image",
                 label_key: str = "label",
                 num_classes: int = 2,
                 ratios: Sequence[float] = None,
                 transform=None,
                 random_pad: bool = False,
                 no_pad: bool = False):
        super().__init__(data_dir=data_dir,
                         json_file=json_file,
                         data_list_key=data_list_key,
                         image_key=image_key,
                         label_key=label_key)

        self.patch_size = (patch_size,) * 3 if isinstance(patch_size, int) else patch_size
        self.patch_batch_size = patch_batch_size
        self.transform = transform
        self.random_pad = random_pad
        self.no_pad = no_pad
        self.cropper = transforms.RandCropByLabelClassesd(
            keys=[image_key, label_key],
            label_key=label_key,
            spatial_size=self.patch_size,
            ratios=ratios if ratios else [1.0] * num_classes,
            num_classes=num_classes,
            num_samples=self.patch_batch_size,
            warn=False,
            allow_smaller=True
        )
        self.cropper.set_random_state(0)

    def __getitem__(self, idx):

        # Keep images in disk memory
        data = self.load_data(idx)

        patches = self.cropper(data)

        # Load only patches to RAM
        for i, patch in enumerate(patches):
            patches[i] = {
                self.image_key: torch.clone(patch[self.image_key]),
                self.label_key: torch.clone(patch[self.label_key])
            }
            # Pad patch if smaller than ROI size
            if not self.no_pad:
                patches[i] = self.pad_patch(patch)

        # Augmentation
        if self.transform:
            patches = self.transform(patches)

        return patches

    def pad_patch(self, patch: dict[str:torch.Tensor, str:torch.Tensor]):
        shapes = zip(patch[self.image_key].shape[::-1], self.patch_size[::-1])
        padding = []
        for shape in shapes:
            pad = shape[1] - shape[0]
            if self.random_pad:
                rand = np.random.randint(pad + 1)
                padding += [rand, pad - rand]
            else:
                padding += [0, pad]

        patch = {
            self.image_key: torch.nn.functional.pad(patch[self.image_key], padding),
            self.label_key: torch.nn.functional.pad(patch[self.label_key], padding)
        }

        return patch


class MultiTaskDataset(Dataset):
    def __init__(self, data_dir: str,
                 json_file: str,
                 data_list_key: str | Sequence[str],
                 pre_image_key: str = "pre_image",
                 post_image_key: str = "post_image",
                 pre_label_key: str = "pre_label",
                 post_label_key: str = "post_label",
                 response_label_key: str = "response",
                 transform=None,
                 load_meta: bool = False):

        self.pre_image_key = pre_image_key
        self.post_image_key = post_image_key
        self.pre_label_key = pre_label_key
        self.post_label_key = post_label_key
        self.response_label_key = response_label_key

        self.data_dir = data_dir
        self.transform = transform
        self.load_meta = load_meta

        self.data_list = []
        json_path = os.path.join(data_dir, json_file)
        with open(json_path, "r") as fp:
            data = json.load(fp)

        if isinstance(data_list_key, (list, tuple)):
            for key in data_list_key:
                if key not in data:
                    raise ValueError(f"key '{key}' not in data")
                self.data_list += data[key]
        else:
            if data_list_key not in data:
                raise ValueError(f"key '{data_list_key}' not in data")
            self.data_list = data[data_list_key]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.load_data(idx)

        # TODO - do augmentation here

        return data

    def load_data(self, idx):
        item = self.data_list[idx]
        pre_image = np.expand_dims(np.load(item[self.pre_image_key]), axis=0)
        post_image = np.expand_dims(np.load(item[self.post_image_key]), axis=0)
        pre_label = np.expand_dims(np.load(item[self.pre_label_key]), axis=0)
        post_label = np.expand_dims(np.load(item[self.post_label_key]), axis=0)

        data = {
            self.pre_image_key: pre_image,
            self.post_image_key: post_image,
            self.pre_label_key: pre_label,
            self.post_label_key: post_label,
            self.response_label_key: item[self.response_label_key]
        }

        return data


def get_intensity_aug(args):
    augmentation_transform = transforms.Compose(
        [
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
                prob=args.rand_contrast_prob,
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=False),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    augmentation_transform.set_random_state(0)
    return augmentation_transform


def get_affine_aug(
        flips=(0.5, 0.5, 0.5),
        theta_x=(-torch.pi, torch.pi),
        theta_y=(-0.1, 0.1),
        theta_z=(-0.1, 0.1),
        scale_x=(0.9, 1.0),
        scale_y=(0.7, 1.1),
        scale_z=(0.7, 1.1),
        shear_yz=(-0.15, 0.15),
        shear_xy=(-0.10, 0.10),
        shear_xz=(-0.10, 0.10),
        theta_probs=(0.2, 0.2, 0.2),
        scale_probs=(0.2, 0.2, 0.2),
        shear_probs=(0.2, 0.2, 0.2),
):
    affine_transform = RandAffineTransformd(
        keys=["image", "label"],
        mode=["bilinear", "nearest"],
        flips=flips,
        thetas=[theta_x,
                theta_y,
                theta_z],
        axes=[(0, 1),
              (0, 2),
              (1, 2)],
        scales=[scale_z,
                scale_y,
                scale_x],
        shears=[shear_yz,
                shear_xy,
                shear_xz],
        theta_probs=theta_probs,
        scale_probs=scale_probs,
        shear_probs=shear_probs,
    )
    return affine_transform


def main():
    from monai.data import DataLoader
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = {
        1: ListedColormap(["red"]),
        2: ListedColormap(["blue"]),
    }

    AXIS = 2

    config_map = {
        0: {"aspect": 1.0, "index": 16, "figsize": (8, 8)},
        1: {"aspect": 5.0 / 0.8, "index": 128, "figsize": (10, 8)},
        2: {"aspect": 5.0 / 0.8, "index": 128, "figsize": (10, 8)},
    }

    CONFIG = config_map[AXIS]
    print(CONFIG)

    transform = get_affine_aug(
        flips=(0.5, 0.5, 0.5),
        scale_x=(0.9, 1.0),
        scale_y=(0.7, 1.0),
        scale_z=(0.7, 1.0),
        theta_x=(-3.14, 3.14),
        theta_y=(-0.1, 0.1),
        theta_z=(-0.1, 0.1),
        shear_yz=(-0.15, 0.15),
        shear_xy=(-0.1, 0.1),
        shear_xz=(-0.1, 0.1),
    )

    def plot_image_label(image, label, ax):
        ax.imshow(image, cmap='gray')

        for key, color in cmap.items():
            ax.imshow(label == key, cmap=cmap[key], alpha=0.4 * (label == key), aspect=CONFIG["aspect"])

        ax.set_xticks([])
        ax.set_yticks([])

    dataset = SegmentationPatchDataset(
        data_dir="numpys_2",
        json_file="data.json",
        data_list_key="training",
        patch_size=(32, 256, 256),
        patch_batch_size=3,
        num_classes=3,
        no_pad=True
    )

    dataloader = DataLoader(dataset, batch_size=1)

    fig, axs = plt.subplots(3, 3, figsize=CONFIG["figsize"])
    for i, batch in enumerate(dataloader):
        batch = transform(batch)

        images, targets = batch["image"], batch["label"]
        print(images.shape)

        for j, (image, label) in enumerate(zip(images, targets)):
            img_slice = np.take(image.squeeze(), indices=CONFIG["index"], axis=AXIS)
            seg_slice = np.take(label.squeeze(), indices=CONFIG["index"], axis=AXIS)

            plot_image_label(img_slice, seg_slice, axs[i, j])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
