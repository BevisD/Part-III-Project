import torch
from torch.cuda.amp import GradScaler, autocast
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from utils import SegmentationPatchDataset, SegmentationDataset, get_intensity_aug, SwinInferer
from optimizers import LinearWarmupCosineAnnealingLR

DATA_DIR = "/bask/projects/p/phwq4930-gbm/Bevis/data/numpy/NeOv_omentum_cropped"
JSON_FILE = "data-small.json"
WORKERS = 0
SW_BATCH = 2
ROI_SIZE = (32, 256, 256)
EPOCHS = 1

TOTAL_MEM = torch.cuda.memory_allocated()


def format_mem(mem):
    if mem < 1e3:
        return f"{mem:.0f}B"
    elif mem < 1e6:
        return f"{mem // 1e3:.0f}KB"
    elif mem < 1e9:
        return f"{mem // 1e6:.0f}MB"
    elif mem < 1e12:
        return f"{mem // 1e9:.0f}GB"


def print_mem_stats(msg):
    global TOTAL_MEM
    mem = torch.cuda.memory_allocated()
    delta = mem - TOTAL_MEM
    TOTAL_MEM = mem
    if delta > 0:
        print(f"{msg}: {format_mem(mem)}, Change: +{format_mem(delta)}")
    elif delta < 0:
        print(f"{msg}: {format_mem(mem)}, Change: -{format_mem(abs(delta))}")
    else:
        print(f"{msg}: {format_mem(mem)}")


def print_mem_accumulated():
    print(f"Max Allocated: {format_mem(torch.cuda.max_memory_allocated())}")
    print(f"Max Reserved: {format_mem(torch.cuda.max_memory_reserved())}")
    torch.cuda.reset_peak_memory_stats()


torch.cuda.set_device(0)
print_mem_stats("Initial")

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
print_mem_stats("Load Model")

weights = torch.load("pretrained_models/model_swinvit.pt")
print_mem_stats("Load Weights")

model.load_from(weights)
print_mem_stats("Load Weights Into Model")


class NameSpace:
    def __init__(self):
        pass


args = NameSpace()
args.rand_scale_prob = 1.0
args.rand_shift_prob = 1.0
args.rand_noise_prob = 1.0
args.rand_smooth_prob = 1.0
args.rand_contrast_prob = 1.0

augmentation_transform = get_intensity_aug(args)
print_mem_stats("Load Transform")

train_dataset = SegmentationPatchDataset(
    data_dir=DATA_DIR,
    json_file=JSON_FILE,
    data_list_key="training",
    patch_size=ROI_SIZE,
    patch_batch_size=2,
    transform=augmentation_transform,
    no_pad=True
)
print_mem_stats("Load Train Dataset")

val_dataset = SegmentationDataset(
    data_dir=DATA_DIR,
    json_file=JSON_FILE,
    data_list_key="validation"
)
print_mem_stats("Load Val Dataset")

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=WORKERS,
    pin_memory=True,
    shuffle=True
)
print_mem_stats("Load Train Loader")

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    num_workers=WORKERS,
    pin_memory=True
)
print_mem_stats("Load Val Dataset")

# Transforms
post_pred = lambda x: torch.argmax(x, dim=1, keepdim=True)
print_mem_stats("Load Post Pred")

loss_func = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
print_mem_stats("Load Loss")

acc_func = DiceMetric(include_background=False, get_not_nans=True, num_classes=2)
print_mem_stats("Load Metric")

model_inferer = SwinInferer(
    model,
    roi_size=ROI_SIZE,
    sw_batch_size=SW_BATCH
)
print_mem_stats("Load Inferer")

model.cuda(0)
print_mem_stats("Model CUDA")

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1e-4,
                              weight_decay=1e-5)
print_mem_stats("Load Optimizer")

scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_epochs=5,
    max_epochs=10,
    warmup_start_lr=1e-4,
)
print_mem_stats("Load Scheduler")

scaler = GradScaler()
print_mem_stats("Load Scaler")

for _ in range(EPOCHS):
    print("================== TRAIN ===================")
    model.train()
    print_mem_stats("Model Train Mode")
    for index, batch in enumerate(train_loader):
        print(f"BATCH {index}")
        print_mem_stats("Enumerate Batch")
        data, target = batch["image"], batch["label"]
        print_mem_stats("Extract Batch")
        data, target = data.cuda(0), target.cuda(0)
        print_mem_stats("Batch To CUDA")

        with autocast():
            print_mem_stats("Train Autocast")
            optimizer.zero_grad()
            print_mem_stats("Zero Grad")

            logits = model(data)  # B*P 2 X Y Z
            print_mem_stats("Forward Model")

            loss = loss_func(logits, target)  # 0-dim tensor
            print_mem_stats("Calc Loss")

            y_pred = post_pred(logits)
            print_mem_stats("Train Post Pred")
            acc_func.reset()
            print_mem_stats("Train Reset Acc")
            acc_func(y_pred=y_pred, y=target)
            print_mem_stats("Train Calc Acc")
            acc, val_not_nan = acc_func.aggregate()
            print_mem_stats("Train Aggregate")

            scaler.scale(loss).backward()
            print_mem_stats("Scaler Backward")
            scaler.step(optimizer)
            print_mem_stats("Scaler Step")
            scaler.update()
            print_mem_stats("Scaler Update")

        print_mem_accumulated()

    print("================= VALIDATE =================")
    model.eval()
    print_mem_stats("Eval Mode")

    with torch.no_grad():
        print_mem_stats("No Grad")
        for index, batch in enumerate(val_loader):
            print_mem_stats("Val Enumerate Batch")
            data, target = batch["image"], batch["label"]
            print_mem_stats("Val Extract Data")
            data, target = data.cuda(0), target.cuda(0)  # 1 1 H W D
            print_mem_stats("Val To CUDA")

            with autocast():
                print_mem_stats("Val Autocast")
                logits = model_inferer(data)  # 1 C H W D
                print_mem_stats("Val Foward")

                val_loss = loss_func(logits, target)
                print_mem_stats("Val Loss")

                y_pred = post_pred(logits)
                print_mem_stats("Val Post Pred")
                acc_func.reset()
                print_mem_stats("Val Reset Acc")
                acc_func(y_pred=y_pred, y=target)
                print_mem_stats("Val Calc Acc")
                acc, val_not_nan = acc_func.aggregate()
                print_mem_stats("Val Aggregate")

        print_mem_accumulated()
