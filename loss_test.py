import torch
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.transforms import AsDiscrete
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from monai.data.utils import decollate_batch
from trainer import Trainer
from monai.data import DataLoader, Dataset
from utils import SwinInferer

class IdenNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


torch.random.manual_seed(2)
out_channels = 2
batch = 2

trainer = Trainer()
trainer.model = IdenNet()

# trainer.model_inferer = SwinInferer(
#     model=trainer.model,
#     roi_size=7,
#     sw_batch_size=4,
#     overlap=0.5
# )

trainer.loss_func = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
trainer.acc_func = DiceMetric(include_background=False, get_not_nans=True, num_classes=out_channels)

trainer.post_pred = lambda x: torch.argmax(x, dim=1, keepdim=True)
# trainer.post_label = lambda x: torch.squeeze(x, dim=1)

inputs = torch.randn(2, 2, 2, 2, 1)
target = torch.randint(0, 2, (2, 1, 2, 2, 1))

data = [
    {
        "image": inputs[0],
        "label": target[0]
    },
    {
        "image": inputs[1],
        "label": target[1]
    }
]

dataset = Dataset(data)
dataloader = DataLoader(dataset, batch_size=1)

trainer.val_loader = dataloader

val_loss, val_acc = trainer.val_epoch()
print(val_loss, val_acc)
