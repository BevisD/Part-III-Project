import argparse
import warnings

import torch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader

from utils import SwinInferer, SegmentationPatchDataset, SegmentationDataset, get_intensity_aug, get_affine_aug
from trainer import Trainer
from optimizers import LinearWarmupCosineAnnealingLR
from models.anisoswinunetr import AnisoSwinUNETR

parser = argparse.ArgumentParser()
# Model Architecture
parser.add_argument("--roi-size-x", type=int, default=32)
parser.add_argument("--roi-size-y", type=int, default=256)
parser.add_argument("--roi-size-z", type=int, default=256)
parser.add_argument("--patch-size-x", type=int, default=2)
parser.add_argument("--patch-size-y", type=int, default=2)
parser.add_argument("--patch-size-z", type=int, default=2)
parser.add_argument("--window-size-x", type=int, default=7)
parser.add_argument("--window-size-y", type=int, default=7)
parser.add_argument("--window-size-z", type=int, default=7)
parser.add_argument("--in-channels", type=int, default=1)
parser.add_argument("--out-channels", type=int, default=2)
parser.add_argument("--feature-size", type=int, default=48)
parser.add_argument("--load-checkpoint", action="store_true")
parser.add_argument("--overwrite-scheduler", action="store_true")

# Training Hyperparameters
parser.add_argument("--optimizer", type=str, default="adamw")
parser.add_argument("--drop-rate", type=float, default=0.0)
parser.add_argument("--attn-drop-rate", type=float, default=0.0)
parser.add_argument("--path-drop-rate", type=float, default=0.0)
parser.add_argument("--grad-checkpoint", action="store_true")
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--warmup-start-lr", type=float, default=0.0)
parser.add_argument("--end-learning-rate", type=float, default=0.0)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.99)
parser.add_argument("--max-epochs", type=int, required=True)
parser.add_argument("--warmup-epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--sw-batch-size", type=int, default=4)
parser.add_argument("--val-every", type=int, default=4)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--grad-scaler", action="store_true")

# Augmentation
parser.add_argument("--rand-scale-prob", type=float, default=0.15)
parser.add_argument("--rand-shift-prob", type=float, default=0.15)
parser.add_argument("--rand-noise-prob", type=float, default=0.15)
parser.add_argument("--rand-smooth-prob", type=float, default=0.2)
parser.add_argument("--rand-contrast-prob", type=float, default=0.15)
parser.add_argument("--rand-flip-x", type=float, default=0.5)
parser.add_argument("--rand-flip-y", type=float, default=0.5)
parser.add_argument("--rand-flip-z", type=float, default=0.5)
parser.add_argument("--rand-rot-range-x", type=float, default=torch.pi)
parser.add_argument("--rand-rot-range-y", type=float, default=0.1)
parser.add_argument("--rand-rot-range-z", type=float, default=0.1)
parser.add_argument("--rand-rot-prob-x", type=float, default=0.2)
parser.add_argument("--rand-rot-prob-y", type=float, default=0.2)
parser.add_argument("--rand-rot-prob-z", type=float, default=0.2)
parser.add_argument("--rand-scale-low-x", type=float, default=0.9)
parser.add_argument("--rand-scale-low-y", type=float, default=0.7)
parser.add_argument("--rand-scale-low-z", type=float, default=0.7)
parser.add_argument("--rand-scale-high-x", type=float, default=1.0)
parser.add_argument("--rand-scale-high-y", type=float, default=1.1)
parser.add_argument("--rand-scale-high-z", type=float, default=1.1)
parser.add_argument("--rand-scale-prob-x", type=float, default=0.2)
parser.add_argument("--rand-scale-prob-y", type=float, default=0.2)
parser.add_argument("--rand-scale-prob-z", type=float, default=0.2)
parser.add_argument("--rand-shear-range-x", type=float, default=0.15)
parser.add_argument("--rand-shear-range-y", type=float, default=0.10)
parser.add_argument("--rand-shear-range-z", type=float, default=0.10)
parser.add_argument("--rand-shear-prob-x", type=float, default=0.20)
parser.add_argument("--rand-shear-prob-y", type=float, default=0.20)
parser.add_argument("--rand-shear-prob-z", type=float, default=0.20)
parser.add_argument("--orient-axes", type=str, default="HWD",
                    choices=["HWD", "HDW", "DHW", "DWH", "WHD", "WDH"])

# Paths
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--json-file", type=str, required=True)
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--pretrained-path", type=str)
parser.add_argument("--use-ssl-pretrained", action="store_true")


def main(args) -> None:
    torch.random.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    intensity_aug = get_intensity_aug(args)
    affine_aug = get_affine_aug(
        flips=(args.rand_flip_x, args.rand_flip_y, args.rand_flip_z),
        theta_x=(-args.rand_rot_range_x, args.rand_rot_range_x),
        theta_y=(-args.rand_rot_range_y, args.rand_rot_range_y),
        theta_z=(-args.rand_rot_range_z, args.rand_rot_range_z),
        scale_x=(args.rand_scale_low_x, args.rand_scale_high_x),
        scale_y=(args.rand_scale_low_y, args.rand_scale_high_y),
        scale_z=(args.rand_scale_low_z, args.rand_scale_high_z),
        shear_yz=(-args.rand_shear_range_x, args.rand_shear_range_x),
        shear_xy=(-args.rand_shear_range_y, args.rand_shear_range_y),
        shear_xz=(-args.rand_shear_range_z, args.rand_shear_range_z),
        theta_probs=(args.rand_rot_prob_x, args.rand_rot_prob_y, args.rand_rot_prob_z),
        scale_probs=(args.rand_scale_prob_x, args.rand_scale_prob_y, args.rand_scale_prob_z),
        shear_probs=(args.rand_shear_prob_x, args.rand_shear_prob_y, args.rand_shear_prob_z),
    )

    trainer = Trainer(
        log_dir=args.log_dir,
        max_epochs=args.max_epochs,
        val_every=args.val_every,
        grad_scale=args.grad_scaler,
        batch_augmentation=affine_aug,
    )

    # Load model
    model = AnisoSwinUNETR(
        img_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        dropout_path_rate=args.path_drop_rate,
        use_checkpoint=args.grad_checkpoint,
        patch_size=(args.patch_size_x, args.patch_size_y, args.patch_size_z),
        window_size=(args.window_size_x, args.window_size_y, args.window_size_z)
    )

    optim_weights = None
    sched_weights = None
    if args.pretrained_path:
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
                if "optimizer" in weights:
                    optim_weights = weights["optimizer"]
                if "scheduler" in weights:
                    sched_weights = weights["scheduler"]

    # Datasets
    train_dataset = SegmentationPatchDataset(
        data_dir=args.data_dir,
        json_file=args.json_file,
        data_list_key="training",
        patch_size=(args.roi_size_x, args.roi_size_y, args.roi_size_z),
        patch_batch_size=args.sw_batch_size,
        num_classes=args.out_channels,
        transform=intensity_aug,  # Only intensity as affine augmentation done on batch in train loop
        no_pad=True  # No padding needed as patch smaller than image (slight performance increase)
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
    trainer.acc_func = DiceMetric(include_background=False,
                                  get_not_nans=True,
                                  num_classes=args.out_channels,
                                  reduction="mean_batch")

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
    if args.optimizer.lower() == "adamw":
        print("Using AdamW optimizer")
        trainer.optimizer = torch.optim.AdamW(model.parameters(),
                                              lr=args.learning_rate,
                                              weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        print("Using SGD optimizer")
        trainer.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} is not known")

    # Scheduler
    # trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     trainer.optimizer,
    #     T_max=args.max_epochs,
    # )

    trainer.scheduler = LinearWarmupCosineAnnealingLR(
        trainer.optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.warmup_start_lr,
        eta_min=args.end_learning_rate
    )

    if optim_weights is not None:
        trainer.optimizer.load_state_dict(optim_weights)
        print(f"Loading Optimiser: Learning Rate is {trainer.optimizer.param_groups[0]['lr']}")

    if sched_weights is not None:
        if not args.overwrite_scheduler:
            trainer.scheduler.load_state_dict(sched_weights)
            print(f"Loading Scheduler: Next epoch is {trainer.scheduler.last_epoch}")
        else:
            print(f"Not Loading Scheduler Checkpoint")

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    args.orient_axes = " ".join(args.orient_axes).lower()
    main(args)
