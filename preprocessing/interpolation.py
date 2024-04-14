import numpy as np
import os
import glob
import nibabel as nib
from pathlib import Path
import argparse
from tqdm import tqdm
from scipy import ndimage


parser = argparse.ArgumentParser(description="Preprocessing tool to resample images to different voxel spacings")

# Paths
parser.add_argument("--image-dir", default="images", type=str)
parser.add_argument("--label-dir", default="labels", type=str)
parser.add_argument("--data-dir", default="data", type=str)
parser.add_argument("--output-dir", default="preprocessed", type=str)
parser.add_argument("--file-extension", default=".nii.gz", type=str)

# Spacings
parser.add_argument("--old-space-x", type=float, required=True)
parser.add_argument("--old-space-y", type=float, required=True)
parser.add_argument("--old-space-z", type=float, required=True)
parser.add_argument("--new-space-x", type=float, required=True)
parser.add_argument("--new-space-y", type=float, required=True)
parser.add_argument("--new-space-z", type=float, required=True)


def main(args):
    img_folders = [args.image_dir]
    seg_folders = [args.label_dir]
    sub_folders = img_folders + seg_folders

    zooms = (
            args.old_space_x / args.new_space_x,
            args.old_space_y / args.new_space_y,
            args.old_space_z / args.new_space_z,
    )
    print(f"Scaling ratios: ({zooms[0]:.2f}, {zooms[1]:.2f}, {zooms[2]:.2f})")

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
            new_data = data.copy()

            if sub_folder in img_folders:
                # Include all image-specific transformations here
                new_data = ndimage.zoom(new_data, (1.0, *zooms[1:]), order=3)  # Cubic interpolation in-plane
                new_data = ndimage.zoom(new_data, (zooms[0], 1.0, 1.0), order=0)  # Nearest interpolation out-of-plane
            elif sub_folder in seg_folders:
                # Include all label-specific transformations here
                new_data = np.round(new_data)
                new_data = ndimage.zoom(new_data, zooms, order=0)  # Nearest interpolation
                new_data = np.round(new_data).astype(np.int8)

            # SAVE IMAGE HERE
            img.header.set_data_shape(new_data.shape)
            img.header.set_zooms((args.new_spacing_x, args.new_spacing_y, args.new_spacing_z))
            nib.save(
                nib.Nifti1Image(new_data, img.affine, img.header),
                os.path.join(args.output_dir, sub_folder, filename + args.file_extension)
            )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
