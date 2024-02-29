import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from utils import post_pred_transform, SwinInferer, get_train_loader, get_test_loader
from trainer import run_training

IMG_SIZE = 96
IN_CHANNELS = 1
OUT_CHANNELS = 2
FEATURE_SIZE = 48
DROP_RATE = 0.2
ATTN_DROP_RATE = 0.0
DROPOUT_PATH_RATE = 0.0
USE_CHECKPOINT = True
SQUARED_PRED = False
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 100
BATCH_SIZE = 4
VAL_EVERY = 4
A_MIN = -1220
A_MAX = 3500
B_MIN = 0
B_MAX = 1
PIXDIM = (1.5, 1.5, 2.0)
DATA_DIR = ""
JSON_LIST = ""
PRETRAINED_PATH = "pretrained_models/swinunetr.pt"


def main() -> None:
    torch.random.manual_seed(0)

    # Load model
    model = SwinUNETR(img_size=IMG_SIZE,
                      in_channels=IN_CHANNELS,
                      out_channels=OUT_CHANNELS,
                      feature_size=FEATURE_SIZE,
                      drop_rate=DROP_RATE,
                      attn_drop_rate=ATTN_DROP_RATE,
                      dropout_path_rate=DROPOUT_PATH_RATE,
                      use_checkpoint=USE_CHECKPOINT)

    weights = torch.load(PRETRAINED_PATH)
    model.load_state_dict(weights["state_dict"])

    # Data Loaders
    train_loader, test_loader = get_train_loader(
        data_dir=DATA_DIR,
        json_list=JSON_LIST,
        batch_size=BATCH_SIZE,
        space=PIXDIM,
        roi_size=IMG_SIZE,
        a_min=A_MIN,
        a_max=A_MAX,
        b_min=B_MIN,
        b_max=B_MAX
    )

    # Loss
    loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, squared_pred=SQUARED_PRED)

    # Output transformations
    post_label = AsDiscrete(to_onehot=OUT_CHANNELS)
    post_pred = post_pred_transform(OUT_CHANNELS)

    # Metric
    dice_acc = DiceMetric(include_background=True, get_not_nans=True)

    # Model Inferer for evaluation
    model_inferer = SwinInferer(model, roi_size=IMG_SIZE)

    # Print number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    run_training(model=model,
                 train_loader=train_loader,
                 test_loader=test_loader,
                 optimizer=optimizer,
                 loss_func=loss,
                 acc_func=dice_acc,
                 scheduler=scheduler,
                 max_epochs=MAX_EPOCHS,
                 batch_size=BATCH_SIZE,
                 val_every=VAL_EVERY,
                 model_inferer=model_inferer,
                 start_epoch=0,
                 post_label=post_label,
                 post_pred=post_pred)


if __name__ == '__main__':
    main()
