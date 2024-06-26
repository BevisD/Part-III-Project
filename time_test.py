import time

import torch
from torch.cuda.amp import GradScaler, autocast
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from utils import SegmentationPatchDataset, SegmentationDataset, get_intensity_aug, get_affine_aug, SwinInferer
from optimizers import LinearWarmupCosineAnnealingLR

DATA_DIR = "/bask/projects/p/phwq4930-gbm/Bevis/data/numpy/NeOv_omentum_cropped"
JSON_FILE = "data.json"
WORKERS = 2
SW_BATCH = 2
ROI_SIZE = (32, 256, 256)
EPOCHS = 1

torch.cuda.set_device(0)

model = SwinUNETR(
    img_size=ROI_SIZE,
    in_channels=1,
    out_channels=2,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True
)

weights = torch.load("pretrained_models/model_swinvit.pt")

model.load_from(weights)


class NameSpace:
    def __init__(self):
        pass


args = NameSpace()
args.rand_scale_prob = 1.0
args.rand_shift_prob = 1.0
args.rand_noise_prob = 1.0
args.rand_smooth_prob = 1.0
args.rand_contrast_prob = 1.0

intensity_aug = get_intensity_aug(args)
affine_aug = get_affine_aug()

train_dataset = SegmentationPatchDataset(
    data_dir=DATA_DIR,
    json_file=JSON_FILE,
    data_list_key="training",
    patch_size=ROI_SIZE,
    patch_batch_size=2,
    transform=intensity_aug,
    no_pad=True
)

val_dataset = SegmentationDataset(
    data_dir=DATA_DIR,
    json_file=JSON_FILE,
    data_list_key="validation"
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=WORKERS,
    pin_memory=True,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=WORKERS,
    pin_memory=True
)

# Transforms
post_pred = lambda x: torch.argmax(x, dim=1, keepdim=True)

loss_func = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

acc_func = DiceMetric(include_background=False, get_not_nans=True, num_classes=2)

model_inferer = SwinInferer(
    model,
    roi_size=ROI_SIZE,
    sw_batch_size=SW_BATCH
)

model.cuda(0)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4,
                              weight_decay=1e-5)

scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_epochs=5,
    max_epochs=10,
    warmup_start_lr=1e-4,
)

scaler = GradScaler()

epoch_start_time = time.perf_counter()
for _ in range(EPOCHS):
    print("================== TRAIN ===================")
    model.train()

    train_start_time = time.perf_counter()
    batch_start_time = time.perf_counter()
    print(time.ctime())
    for index, batch in enumerate(train_loader):
        print(f"BATCH {index}")
        torch.cuda.synchronize(0)
        print(f"\tEnumerate Data {time.perf_counter() - batch_start_time:.4f}")

        t = time.perf_counter()
        batch = affine_aug(batch)
        torch.cuda.synchronize(0)
        print(f"\tAugment Data {time.perf_counter() - t:.4f}")

        t = time.perf_counter()
        data, target = batch["image"], batch["label"]
        data, target = data.cuda(0), target.cuda(0)
        torch.cuda.synchronize(0)
        print(f"\tProcess Data {time.perf_counter() - t:.4f}")

        t = time.perf_counter()
        with autocast():
            optimizer.zero_grad()
            torch.cuda.synchronize(0)
            print(f"\tZero Grad {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            logits = model(data)  # B*P 2 X Y Z
            torch.cuda.synchronize(0)
            print(f"\tForward Pass {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            loss = loss_func(logits, target)  # 0-dim tensor

            y_pred = post_pred(logits)
            acc_func.reset()
            acc_func(y_pred=y_pred, y=target)
            acc, val_not_nan = acc_func.aggregate()

            torch.cuda.synchronize(0)
            print(f"\tLoss + Acc {time.perf_counter() - t:.4f}s")

            t = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize(0)
            print(f"\tBackward {time.perf_counter() - t:.4f}")

        torch.cuda.synchronize(0)
        print(f"Train Batch Time {time.perf_counter() - batch_start_time:.4f}s")
        batch_start_time = time.perf_counter()

    torch.cuda.synchronize(0)
    print(f"Train Time {time.perf_counter() - train_start_time:.4f}s")
    print(time.ctime())

    val_start_time = time.perf_counter()
    print("================= VALIDATE =================")
    model.eval()
    with torch.no_grad():
        batch_start_time = time.perf_counter()
        t = time.perf_counter()
        for index, batch in enumerate(val_loader):
            data, target = batch["image"], batch["label"]
            data, target = data.cuda(0), target.cuda(0)  # 1 1 H W D

            with autocast():
                logits = model_inferer(data)  # 1 C H W D

                val_loss = loss_func(logits, target)

                y_pred = post_pred(logits)
                acc_func.reset()
                acc_func(y_pred=y_pred, y=target)
                acc, val_not_nan = acc_func.aggregate()

            torch.cuda.synchronize(0)
            print(f"Val Batch Time: {time.perf_counter() - t:.4f}s")
            t = time.perf_counter()

    torch.cuda.synchronize()
    print(f"Val Time {time.perf_counter() - val_start_time:.4f}s")

    print(f"Epoch Time {time.perf_counter() - epoch_start_time:.4f}s")
    epoch_start_time = time.perf_counter()

