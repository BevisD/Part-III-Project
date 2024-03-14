import numpy as np
import os
import glob
import nibabel as nib
from pathlib import Path
from scipy.ndimage import zoom
import argparse
import warnings

parser = argparse.ArgumentParser(description="Preprocessing tool to convert from .nii.gz to .npy")

# Intensity Scaling
parser.add_argument("--a-min", default=-150.0, type=float)
parser.add_argument("--a-max", default=250.0, type=float)
parser.add_argument("--b-min", default=0.0, type=float)
parser.add_argument("--b-max", default=1.0, type=float)
parser.add_argument("--no_clip", action="store_true")

# Image Shape
parser.add_argument("--size-x", default=None, type=int)
parser.add_argument("--size-y", default=None, type=int)
parser.add_argument("--size-z", default=None, type=int)
parser.add_argument("--axes-order", default=None, type=str)

# Paths
parser.add_argument("--pre-dir", default="pre_treatment", type=str)
parser.add_argument("--post-dir", default="post_treatment", type=str)
parser.add_argument("--image-dir", default="images", type=str)
parser.add_argument("--label-dir", default="labels", type=str)
parser.add_argument("--data-dir", default="data", type=str)
parser.add_argument("--output-dir", default="preprocessed", type=str)
parser.add_argument("--file-extension", default=".nii.gz", type=str)


def resample_3d(img, target_size):
    for t in target_size:
        if t is None:
            continue

        if t <= 0:
            raise ValueError("Can't scale to non-positive dimension")

    zoom_ratio = [float(t) / float(im) if t is not None else 1
                  for im, t in zip(img.shape, target_size)]
    img_resampled = zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def scale_intensity(data: np.ndarray, a_min: float, a_max: float,
                    b_min: float = 0, b_max: float = 1,
                    clip: bool = True):
    if clip:
        data[data < a_min] = a_min
        data[data > a_max] = a_max

    a_range = a_max - a_min
    b_range = b_max - b_min

    if a_range == 0:
        raise ValueError(f"a_min == a_max is not scalable")

    data = (data - a_min) * b_range / a_range + b_min
    return data


def select_channels(data: np.ndarray, channels: list):
    mask = np.isin(data, channels)
    return data * mask


def main(args):
    target_size = (args.size_x, args.size_y, args.size_z)

    ornt = None
    if args.axes_order is not None:
        ornt = nib.orientations.axcodes2ornt(args.axes_order)
        if any([s is not None for s in target_size]):
            warnings.warn(f"Applying orientation {args.axes_order} before dimension scaling")

    img_folders = [
        os.path.join(args.pre_dir, args.image_dir),
        os.path.join(args.post_dir, args.image_dir),
    ]

    seg_folders = [
        os.path.join(args.pre_dir, args.label_dir),
        os.path.join(args.post_dir, args.label_dir),
    ]

    sub_folders = img_folders + seg_folders

    for sub_folder in sub_folders:
        os.makedirs(os.path.join(args.output_dir, sub_folder), exist_ok=True)

    for sub_folder in sub_folders:
        filepaths = glob.glob(os.path.join(args.data_dir, sub_folder, f"*{args.file_extension}"))

        for filepath in filepaths:
            print(f"Preprocessing {filepath}")
            filename = Path(filepath).stem.split(".")[0]

            img = nib.load(filepath)
            data = img.get_fdata()
            if ornt is not None:
                data = nib.apply_orientation(data, ornt)

            data = resample_3d(data, target_size)

            if sub_folder in img_folders:
                data = scale_intensity(data, args.a_min, args.a_max,
                                       args.b_min, args.b_max, clip=not args.no_clip)
            elif sub_folder in seg_folders:
                data = select_channels(data, channels=[1])
            np.save(os.path.join(args.output_dir, sub_folder, filename), data)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
