import torch
from monai import transforms
from typing import Sequence

__all__ = [
    "SelectChannelTransformd",
    "AddChanneld",
    "compose_test_transform",
    "compose_train_transform"
]


class SelectChannelTransformd(transforms.MapTransform):
    """
    Transformation object that only sets every value to 0 except for
    values within channels
    """
    def __init__(self, keys: Sequence[str],
                 channels: Sequence[int] | int,
                 allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.channels = torch.Tensor(channels)

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                mask = torch.isin(data[key], self.channels)
                data[key] = mask * data[key]
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return data


class AddChanneld(transforms.MapTransform):
    def __init__(self, keys: list[str], allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        for key in self.keys:
            if key in data:
                data[key] = data[key].view(1, *data[key].shape)
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return data


def compose_train_transform(space: float | Sequence[float],
                            a_min: float,
                            a_max: float,
                            b_min: float,
                            b_max: float,
                            roi_size: float | Sequence[int],
                            flip_prob: float = 0.2,
                            rotate_prob: float = 0.2,
                            intensity_scale_prob: float = 0.1,
                            intensity_shift_prob: float = 0.1) -> transforms.Transform:
    """
    Returns the transformation for the training data, augmentation included.

    :param space: PixDim values to scale to
    :param a_min: Minimum intensity of original image
    :param a_max: Maximum intensity of original image
    :param b_min: Minimum intensity of transformed image
    :param b_max: Maximum intensity of transformed image
    :param roi_size: Size of the region of interest to crop to
    :param flip_prob: Probability of randomly flipping independently in each axis
    :param rotate_prob: Probability of randomly rotating 90deg in x-y plane
    :param intensity_scale_prob: Probability of scaling the intensity
    :param intensity_shift_prob: Probability of shifting the intensity
    :return: Transformation object for training data
    """

    # img_keys = ["pre-image", "post-image"]
    # seg_keys = ["pre-label", "post-label"]
    img_keys = ["image"]
    seg_keys = ["label"]
    all_keys = img_keys + seg_keys

    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys),
            SelectChannelTransformd(keys=seg_keys, channels=[1]),
            AddChanneld(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RAS"),
            transforms.Spacingd(keys=all_keys,
                                pixdim=space,
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=img_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            # CROP HERE?
            transforms.RandSpatialCropSamplesd(
                keys=all_keys,
                roi_size=roi_size,
                random_size=False,
                num_samples=4,
            ),
            transforms.RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=0),
            transforms.RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=1),
            transforms.RandFlipd(keys=all_keys, prob=flip_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=all_keys, prob=rotate_prob, max_k=3),
            transforms.RandScaleIntensityd(keys=img_keys, factors=0.1, prob=intensity_scale_prob),
            transforms.RandShiftIntensityd(keys=img_keys, offsets=0.1, prob=intensity_shift_prob),
            transforms.ToTensord(keys=all_keys),
        ]
    )
    return transform


def compose_test_transform(space: Sequence[float] | float,
                           a_max: float,
                           a_min: float,
                           b_min: float,
                           b_max: float) -> transforms.Transform:
    """
    Returns the transformation for the test data.

    :param space: PixDim values to scale to
    :param a_min: Minimum intensity of original image
    :param a_max: Maximum intensity of original image
    :param b_min: Minimum intensity of transformed image
    :param b_max: Maximum intensity of transformed image
    :return: Transformation object for test data
    """

    # img_keys = ["pre-image", "post-image"]
    # seg_keys = ["pre-label", "post-label"]
    img_keys = ["image"]
    seg_keys = ["label"]
    all_keys = img_keys + seg_keys

    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=all_keys),
            SelectChannelTransformd(keys=seg_keys, channels=[1]),
            AddChanneld(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RAS"),
            transforms.Spacingd(keys=all_keys,
                                pixdim=space,
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(
                keys=img_keys, a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            ),
            # CROP HERE?
            transforms.ToTensord(keys=all_keys),
        ]
    )
    return transform
