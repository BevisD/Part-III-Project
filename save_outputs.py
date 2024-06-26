import argparse
import csv
import os
import pathlib
import pickle

import nibabel as nib
import numpy as np
import torch
from torch.cuda.amp import autocast
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader
from monai.metrics import DiceMetric

from utils import SegmentationDataset, SwinInferer

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--json-file", type=str, required=True)
parser.add_argument("--pretrained-path", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--meta-file", type=str, default="meta.pkl")
parser.add_argument("--data-list-key", type=str, default="validation")
parser.add_argument("--image-key", type=str, default="image")
parser.add_argument("--label-key", type=str, default="label")
parser.add_argument("--save-outputs", action="store_true")

parser.add_argument("--roi-size-x", type=int, default=32)
parser.add_argument("--roi-size-y", type=int, default=256)
parser.add_argument("--roi-size-z", type=int, default=256)
parser.add_argument("--in-channels", type=int, default=1)
parser.add_argument("--out-channels", type=int, default=2)
parser.add_argument("--feature-size", type=int, default=48)
parser.add_argument("--sw-batch-size", type=int, default=4)
parser.add_argument("--infer-overlap", type=float, default=0.5)

parser.add_argument("--workers", type=int, default=0)


def main(args):
    model = SwinUNETR(
        img_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size
    )

    weights = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(weights["state_dict"])

    test_dataset = SegmentationDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        data_list_key=args.data_list_key,
        image_key=args.image_key,
        label_key=args.label_key,
        load_meta=True,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=args.workers,
        batch_size=1,
        pin_memory=True
    )

    model_inferer = SwinInferer(
        model,
        roi_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        sw_batch_size=args.sw_batch_size,
        overlap=args.infer_overlap
    )
    model.cuda(0)

    acc_func = DiceMetric(include_background=False, get_not_nans=True, num_classes=args.out_channels)
    post_pred = lambda x: torch.argmax(x, dim=1, keepdim=True)

    header, affine = None, None
    if args.save_outputs:
        with open(os.path.join(args.data_dir, args.meta_file), "rb") as file:
            pickle_dict = pickle.load(file)
            header = pickle_dict["header"]
            affine = pickle_dict["affine"]

    results = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            data, target, filename = batch["image"], batch["label"], batch["filename"][0]
            # data  1 1 H W D - torch.HalfTensor
            # label 1 1 H W D - torch.CharTensor
            # filename - str
            data, target = data.cuda(0), target.cuda(0)

            print(f"Inference on {filename}\t", end="")
            with autocast():
                logits = model_inferer(data)  # 1 2 H W D
            out = post_pred(logits)

            acc_func.reset()
            acc_func(y_pred=out, y=target)
            acc, not_nan = acc_func.aggregate()

            results.append({
                "Filename": filename,
                "Accuracy": acc.item(),
                "NaN": not bool(not_nan)
            })

            print(f"Accuracy: {acc.item():.5f}{'' if not_nan else ' NaN'}")

            # Save to .nii.gz
            if args.save_outputs:
                file_path = pathlib.Path(filename)
                nii_filename = file_path.parent / (file_path.stem + ".nii.gz")
                (args.output_dir / pathlib.Path(filename)).parent.mkdir(parents=True, exist_ok=True)
                nib.save(
                    nib.Nifti1Image(out.cpu().numpy().astype(np.int8)[0, 0], affine, header),
                    os.path.join(args.output_dir, nii_filename)
                )

    # Save to .csv
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, "accuracies.csv"), "w+") as file:
        writer = csv.DictWriter(file, fieldnames=["Filename", "Accuracy", "NaN"])
        writer.writeheader()

        for row in results:
            writer.writerow(row)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
