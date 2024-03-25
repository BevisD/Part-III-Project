import nibabel as nib
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--from-dir", type=str, required=True)
parser.add_argument("--to-dir", type=str, required=True)
parser.add_argument("--image-dir", type=str, default="images")
parser.add_argument("--label-dir", type=str, default="labels")


def copy_img(filename, from_dir, to_dir, sub_folder="images"):
    file = nib.load(os.path.join(from_dir, sub_folder, filename))

    data = file.get_fdata()
    pre, post = data

    pre_img = nib.Nifti1Image(pre, file.affine, file.header)
    post_img = nib.Nifti1Image(post, file.affine, file.header)

    nib.save(pre_img, os.path.join(to_dir, "pre_treatment", sub_folder, filename))
    nib.save(post_img, os.path.join(to_dir, "post_treatment", sub_folder, filename))


def main(args):
    image_names = os.listdir(os.path.join(args.from_dir, args.image_dir))
    label_names = os.listdir(os.path.join(args.from_dir, args.label_dir))

    for i, image_name in enumerate(image_names):
        print(f"Copying image {image_name} {i}/{len(image_names)}")
        copy_img(filename=image_name,
                 from_dir=args.from_dir,
                 to_dir=args.to_dir,
                 sub_folder=args.image_dir
                 )

    for i, label_name in enumerate(label_names):
        print(f"Copying label {label_name} {i}/{len(label_names)}")
        copy_img(filename=label_name,
                 from_dir=args.from_dir,
                 to_dir=args.to_dir,
                 sub_folder=args.label_dir
                 )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
