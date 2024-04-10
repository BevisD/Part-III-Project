import argparse
import json
import os

import numpy as np
import nibabel as nib
from skimage import measure
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="")
parser.add_argument("--connectivity", type=int, default=3)


def numpy_to_nifti(numpy_path):
    numpy_path = numpy_path.replace("numpy", "nifti")
    numpy_path = numpy_path.replace(".npy", ".nii.gz")
    return numpy_path


def component_volumes(label, vol_per_vox, args):
    volume_dict = {}

    component_map, num_components = measure.label(
        label,
        connectivity=args.connectivity,
        return_num=True
    )

    volumes = [np.count_nonzero(component_map == lesion_num) * vol_per_vox for lesion_num in range(1, num_components+1)]
    volumes.sort(reverse=True)

    for lesion_num, volume in enumerate(volumes, start=1):
        volume_dict[f"lesion{lesion_num}"] = volume

    return volume_dict


def main(args):
    # Get filenames to analyse
    label_dir = os.path.join(args.data_dir, "labels")
    label_files = os.listdir(label_dir)

    component_data = {}
    # Load segmentation
    for file in tqdm(label_files):
        component_data[file] = {}

        nib_label = nib.load(os.path.join(label_dir, file))
        label_data = np.round(nib_label.get_fdata()).squeeze()
        vol_per_vox = np.prod(nib_label.header.get_zooms())

        component_data[file]["OMTM"] = component_volumes((label_data == 1).astype(int), vol_per_vox, args)
        component_data[file]["POD"] = component_volumes((label_data == 9).astype(int), vol_per_vox, args)

    with open(os.path.join(args.data_dir, "legions.json"), "w+") as fp:
        print("Saving component data")
        json.dump(component_data, fp, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
