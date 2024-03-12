import argparse

import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from utils import post_pred_transform, SwinInferer, get_train_loader
from trainer import run_training

parser = argparse.ArgumentParser()
# Model Architecture
parser.add_argument("--roi-size", type=int, default=96)
parser.add_argument("--in-channels", type=int, default=1)
parser.add_argument("--out-channels", type=int, default=2)
parser.add_argument("--feature-size", type=int, default=48)

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
parser.add_argument("--val-every", type=int, default=4)

# Transform Parameters
parser.add_argument("--a-min", type=float, default=-1220)
parser.add_argument("--a-max", type=float, default=3500)
parser.add_argument("--b-min", type=float, default=0)
parser.add_argument("--b-max", type=float, default=1)
parser.add_argument("--space-x", type=float, default=1.0)
parser.add_argument("--space-y", type=float, default=1.0)
parser.add_argument("--space-z", type=float, default=1.0)

# Paths
parser.add_argument("--data-dir", type=str, required=True)
parser.add_argument("--json-list", type=str, required=True)
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--pretrained-path", type=str, required=True)


def main(args) -> None:
    torch.random.manual_seed(0)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    # Load model
    model = SwinUNETR(img_size=args.roi_size,
                      in_channels=args.in_channels,
                      out_channels=args.out_channels,
                      feature_size=args.feature_size,
                      drop_rate=args.drop_rate,
                      attn_drop_rate=args.attn_drop_rate,
                      dropout_path_rate=args.path_drop_rate,
                      use_checkpoint=args.grad_checkpoint)

    weights = torch.load(args.pretrained_path)
    model.load_state_dict(weights["state_dict"])

    # Data Loaders
    train_loader, test_loader = get_train_loader(
        data_dir=args.data_dir,
        json_list=args.json_list,
        batch_size=args.batch_size,
        space=(args.space_x, args.space_y, args.space_z),
        roi_size=args.roi_size,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max
    )

    # Loss
    loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, squared_pred=args.square_pred)

    # Output transformations
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = post_pred_transform(args.out_channels)

    # Metric
    dice_acc = DiceMetric(include_background=False)

    # Model Inferer for evaluation
    model_inferer = SwinInferer(model, roi_size=args.roi_size)

    # Print number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model.cuda(0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    run_training(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_func=loss,
        acc_func=dice_acc,
        scheduler=scheduler,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        val_every=args.val_every,
        model_inferer=model_inferer,
        start_epoch=0,
        post_label=post_label,
        post_pred=post_pred,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
