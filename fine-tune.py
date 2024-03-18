import argparse

import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader

from utils import post_pred_transform, SwinInferer, SegmentationPatchDataset, SegmentationDataset, \
    get_augmentation_transform
from trainer import Trainer

parser = argparse.ArgumentParser()
# Model Architecture
parser.add_argument("--roi-size", type=int, default=96)
parser.add_argument("--in-channels", type=int, default=1)
parser.add_argument("--out-channels", type=int, default=2)
parser.add_argument("--feature-size", type=int, default=48)
parser.add_argument("--load-checkpoint", action="store_true")

# Training Hyperparameters
parser.add_argument("--drop-rate", type=float, default=0.0)
parser.add_argument("--attn-drop-rate", type=float, default=0.0)
parser.add_argument("--path-drop-rate", type=float, default=0.0)
parser.add_argument("--grad-checkpoint", action="store_true")
parser.add_argument("--square-pred", action="store_true")
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--max-epochs", type=int, required=True)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--sw-batch-size", type=int, default=4)
parser.add_argument("--val-every", type=int, default=4)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--rand-flip-prob", type=float, default=0.2)
parser.add_argument("--rand-rot-prob", type=float, default=0.2)
parser.add_argument("--rand-scale-prob", type=float, default=0.1)
parser.add_argument("--rand-shift-prob", type=float, default=0.1)
parser.add_argument("--rand-noise-prob", type=float, default=0.0)
parser.add_argument("--grad-scaler", action="store_true")

# Paths
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--json-file", type=str, required=True)
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--pretrained-path", type=str, required=True)


def main(args) -> None:
    torch.random.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    augmentation_transform = get_augmentation_transform(args)

    trainer = Trainer(
        log_dir=args.log_dir,
        max_epochs=args.max_epochs,
        val_every=args.val_every
    )

    # Load model
    model = SwinUNETR(
        img_size=args.roi_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.path_drop_rate,
        use_checkpoint=args.grad_checkpoint
    )

    weights = torch.load(args.pretrained_path)
    model.load_state_dict(weights["state_dict"])

    if args.load_checkpoint:
        if "epoch" in weights:
            trainer.start_epoch = weights["epoch"]
        if "best_acc" in weights:
            trainer.best_val_acc = weights["best_acc"]
        print(f"Resuming training from epoch {trainer.start_epoch} best-acc {trainer.best_val_acc:.5f}")

    # Datasets
    train_dataset = SegmentationPatchDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        data_list_key="training",
        patch_size=args.roi_size,
        patch_batch_size=args.sw_batch_size,
        transform=augmentation_transform
    )

    val_dataset = SegmentationDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        data_list_key="validation"
    )

    # Data loaders
    trainer.train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True
    )
    trainer.val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True
    )

    # Transforms
    trainer.post_pred = post_pred_transform(threshold=0.5)

    # Loss
    trainer.loss_func = DiceLoss(sigmoid=True)

    # Metric
    trainer.acc_func = DiceMetric(get_not_nans=True)

    # Model Inferer for evaluation
    trainer.model_inferer = SwinInferer(model, roi_size=args.roi_size)

    # Print number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model.cuda(0)
    trainer.model = model

    # Optimizer
    trainer.optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

    # Scheduler
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.max_epochs)

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
