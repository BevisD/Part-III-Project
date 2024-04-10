import argparse
import csv
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.data import DataLoader
from skimage import measure

from utils import SwinInferer, SegmentationDataset, get_intensity_aug, get_affine_aug

parser = argparse.ArgumentParser()
# Paths
parser.add_argument("--data-dir", type=str, default="")
parser.add_argument("--json-file", type=str, default="data.json")
parser.add_argument("--data-list-key", type=str, default="validation")
parser.add_argument("--pretrained-path", type=str, required=True)
parser.add_argument("--output-csv", type=str, required=True)

# Model Architecture
parser.add_argument("--roi-size-x", type=int, default=32)
parser.add_argument("--roi-size-y", type=int, default=256)
parser.add_argument("--roi-size-z", type=int, default=256)
parser.add_argument("--in-channels", type=int, default=1)
parser.add_argument("--out-channels", type=int, default=3)
parser.add_argument("--feature-size", type=int, default=48)
parser.add_argument("--sw-batch-size", type=int, default=4)
parser.add_argument("--workers", type=int, default=0)

# Augmentation
parser.add_argument("--augmentation", action="store_true")
parser.add_argument("--intersection", action="store_true")
parser.add_argument("--rand-scale-prob", type=float, default=0.15)
parser.add_argument("--rand-shift-prob", type=float, default=0.15)
parser.add_argument("--rand-noise-prob", type=float, default=0.15)
parser.add_argument("--rand-smooth-prob", type=float, default=0.2)
parser.add_argument("--rand-contrast-prob", type=float, default=0.15)


def get_intersecting_components(label: torch.Tensor, pred: torch.Tensor):
    assert label.ndim == pred.ndim == 5
    assert label.shape[0] == pred.shape[0] == 1
    assert label.shape[1] == pred.shape[1] == 1

    device = pred.device

    label = label.squeeze().cpu()
    pred = pred.squeeze().cpu()

    omtm_pred_lesions = torch.Tensor(measure.label(pred == 1))
    pod_pred_lesions = torch.Tensor(measure.label(pred == 2))

    omtm_target_mask = (label == 1)
    pod_target_mask = (label == 2)

    omtm_intersect = torch.unique(omtm_target_mask * omtm_pred_lesions)
    omtm_intersect = omtm_intersect[omtm_intersect.nonzero()]

    pod_intersect = torch.unique(pod_target_mask * pod_pred_lesions)
    pod_intersect = pod_intersect[pod_intersect.nonzero()]

    omtm_mask = torch.isin(omtm_pred_lesions, omtm_intersect)
    pod_mask = torch.isin(pod_pred_lesions, pod_intersect)

    pred_mask = omtm_mask + pod_mask

    return (pred * pred_mask).unsqueeze(0).unsqueeze(0).to(device)


def main(args):
    affine_aug = get_affine_aug()
    intensity_aug = get_intensity_aug(args)

    # Load Model
    model = SwinUNETR(
        img_size=args.roi_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size
    )
    weights = torch.load(args.pretrained_path)
    model.load_state_dict(weights["state_dict"])

    # Create datasets/dataloader
    test_dataset = SegmentationDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        data_list_key=args.data_list_key,
        transform=intensity_aug,
        load_meta=True
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
    )
    model.cuda(0)

    acc_func = DiceMetric(
        include_background=False,
        get_not_nans=True,
        num_classes=args.out_channels,
        reduction="mean_batch"
    )

    results = []
    with torch.no_grad():
        total_cases = len(test_loader)
        for i, batch in enumerate(test_loader):
            if args.augmentation:
                batch = affine_aug(batch)

            data, target, filename = batch["image"], batch["label"], batch["filename"]
            data, target = data.cuda(0), target.cuda(0)

            print(f"{i}/{total_cases}: {filename}", flush=True)

            with autocast(dtype=torch.bfloat16):
                logits = model_inferer(data)

            pred = torch.argmax(logits, dim=1, keepdim=True)

            if args.intersection:
                pred = get_intersecting_components(target, pred)

            acc_func.reset()
            acc_func(y=target, y_pred=pred)
            acc, not_nan = acc_func.aggregate()

            omtm_acc, pod_acc = acc.tolist()
            omtm_nan, pod_nan = acc.logical_not().tolist()

            results.append({
                "filename": filename,
                "omtm-acc": omtm_acc,
                "pod-acc": pod_acc,
                "omtm-nan": omtm_nan,
                "pod-nan": pod_nan
            })

    out_dir = Path(args.pretrained_path).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / args.output_csv
    print("Saving Results")
    with open(out_path, "w+") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "omtm-acc", "pod-acc", "omtm-nan", "pod-nan"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    args = parser.parse_args()
    args.roi_size = (args.roi_size_x, args.roi_size_y, args.roi_size_z)
    main(args)
