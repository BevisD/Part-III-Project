import argparse
import warnings

import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader

from utils import SwinInferer, SegmentationPatchDataset, SegmentationDataset, get_intensity_aug, get_affine_aug
from trainer import Trainer
from optimizers import LinearWarmupCosineAnnealingLR

parser = argparse.ArgumentParser()
# Model Architecture
parser.add_argument("--roi-size-x", type=int, default=32)
parser.add_argument("--roi-size-y", type=int, default=256)
parser.add_argument("--roi-size-z", type=int, default=256)
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
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--max-epochs", type=int, required=True)
parser.add_argument("--warmup-epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--sw-batch-size", type=int, default=4)
parser.add_argument("--val-every", type=int, default=4)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--rand-scale-prob", type=float, default=0.15)
parser.add_argument("--rand-shift-prob", type=float, default=0.15)
parser.add_argument("--rand-noise-prob", type=float, default=0.15)
parser.add_argument("--rand-smooth-prob", type=float, default=0.2)
parser.add_argument("--rand-contrast-prob", type=float, default=0.15)
parser.add_argument("--grad-scaler", action="store_true")

# Paths
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--json-file", type=str, required=True)
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--pretrained-path", type=str, required=True)
parser.add_argument("--use-ssl-pretrained", action="store_true")


def main(args) -> None:
    torch.random.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    intensity_transform = get_intensity_aug(args)
    affine_transform = get_affine_aug(args)

    trainer = Trainer(
        log_dir=args.log_dir,
        max_epochs=args.max_epochs,
        val_every=args.val_every,
        grad_scale=args.grad_scaler,
        batch_augmentation=affine_transform
    )

    # Load model
    model = SwinUNETR(
        img_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.path_drop_rate,
        use_checkpoint=args.grad_checkpoint
    )
    weights = torch.load(args.pretrained_path)

    if args.use_ssl_pretrained:
        # Use self supervised weights (SwinViT weights only)
        if args.load_checkpoint:
            warnings.warn("Using SSL pretrained: --load-checkpoint ignored")
        else:
            print("Using SSL pretrained")
        model.load_from(weights)
    else:
        # Use saved weight dict (SwinUNETR weights only)
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
        patch_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        patch_batch_size=args.sw_batch_size,
        transform=intensity_transform,
        no_pad=True
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
    trainer.post_pred = lambda x: torch.argmax(x, dim=1, keepdim=True)

    # Loss
    trainer.loss_func = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

    # Metric
    trainer.acc_func = DiceMetric(include_background=False, get_not_nans=True, num_classes=args.out_channels)

    # Model Inferer for evaluation
    trainer.model_inferer = SwinInferer(
        model,
        roi_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        sw_batch_size=args.sw_batch_size
    )

    # Print number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model.cuda(0)
    trainer.model = model

    # Optimizer
    trainer.optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay)

    # trainer.optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.learning_rate,
    #     momentum=args.momentum,
    #     nesterov=True,
    #     weight_decay=args.weight_decay
    # )

    # Scheduler
    # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     trainer.optimizer,
    #     T_max=args.max_epochs,
    # )
    trainer.scheduler = LinearWarmupCosineAnnealingLR(
        trainer.optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.learning_rate,
    )

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
