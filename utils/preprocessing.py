import numpy as np
import os
import glob
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt


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


def main():
    data_dir = "data/NeOv_rigid_sample"
    preprocessed_dir = "preprocessed"

    a_min = -150
    a_max = 250
    b_min = 0
    b_max = 1

    img_folders = [
        "pre_treatment/images",
        "post_treatment/images",
    ]

    seg_folders = [
        "pre_treatment/labels",
        "post_treatment/labels",
    ]

    sub_folders = img_folders + seg_folders

    if not os.path.exists(preprocessed_dir):
        print("Creating Preprocessing Folders")
        os.mkdir(preprocessed_dir)
        os.mkdir(os.path.join(preprocessed_dir, "pre_treatment"))
        os.mkdir(os.path.join(preprocessed_dir, "post_treatment"))

        for sub_folder in sub_folders:
            os.mkdir(os.path.join(preprocessed_dir, sub_folder))

    for sub_folder in sub_folders:
        filepaths = glob.glob(os.path.join(data_dir, sub_folder, "*.nii.gz"))

        for filepath in filepaths:
            print(f"Preprocessing {filepath}")
            filename = Path(filepath).stem.split(".")[0]

            img = nib.load(filepath)
            data = img.get_fdata()

            if sub_folder in img_folders:
                data = scale_intensity(data, a_min, a_max, b_min, b_max, clip=True)
            elif sub_folder in seg_folders:
                data = select_channels(data, channels=[1])
            np.save(os.path.join(preprocessed_dir, sub_folder, filename), data)


if __name__ == '__main__':
    main()
