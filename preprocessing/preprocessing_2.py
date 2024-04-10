import numpy as np
import os
import glob
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Preprocessing tool to convert from .nii.gz to .npy")

# Intensity Scaling
parser.add_argument("--a-min", default=-150.0, type=float)
parser.add_argument("--a-max", default=250.0, type=float)
parser.add_argument("--b-min", default=0.0, type=float)
parser.add_argument("--b-max", default=1.0, type=float)
parser.add_argument("--no_clip", action="store_true")

# Paths
parser.add_argument("--image-dir", default="images", type=str)
parser.add_argument("--label-dir", default="labels", type=str)
parser.add_argument("--data-dir", default="data", type=str)
parser.add_argument("--output-dir", default="preprocessed", type=str)
parser.add_argument("--file-extension", default=".nii.gz", type=str)


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


def select_channels(data: np.ndarray, channels: list, channels_map: dict = None):
    mask = np.isin(data, channels)
    masked_data = mask * data
    new_data = masked_data.copy()

    if channels_map is not None:
        for in_c, out_c in channels_map.items():
            new_data[masked_data == in_c] = out_c
    return new_data


def main(args):
    img_folders = [args.image_dir]
    seg_folders = [args.label_dir]
    sub_folders = img_folders + seg_folders

    for sub_folder in sub_folders:
        os.makedirs(os.path.join(args.output_dir, sub_folder), exist_ok=True)

    for sub_folder in sub_folders:
        filepaths = glob.glob(os.path.join(args.data_dir, sub_folder, f"*{args.file_extension}"))
        print(sub_folder)
        for filepath in tqdm(filepaths):
            # print(f"Preprocessing {filepath}")
            filename = Path(filepath).stem.split(".")[0]

            img = nib.load(filepath)
            data = img.get_fdata().squeeze()

            if sub_folder in img_folders:
                # Include all image-specific transformations here
                data = scale_intensity(data, args.a_min, args.a_max,
                                       args.b_min, args.b_max, clip=not args.no_clip).astype(np.float16)
            elif sub_folder in seg_folders:
                # Include all label-specific transformations here
                data = np.round(data)
                data = select_channels(data, channels=[1, 9], channels_map={1: 1, 9: 2}).astype(np.int8)
            np.save(os.path.join(args.output_dir, sub_folder, filename), data)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
