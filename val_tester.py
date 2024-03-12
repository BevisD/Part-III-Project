from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.networks.utils import one_hot
from monai.data import decollate_batch
from utils import post_pred_transform
import torch

torch.random.manual_seed(0)

B, C, H, W, D = 10, 2, 3, 3, 3

dice_acc = DiceMetric(include_background=True, get_not_nans=True)

post_label = AsDiscrete(to_onehot=C)
post_pred = post_pred_transform(C)

target = torch.randint(0, 2, (B, 1, H, W, D)).to(torch.int64) * 0
logits = torch.randn((B, C, H, W, D))

test_labels_list = decollate_batch(target)
test_outputs_list = decollate_batch(logits)

val_labels_converted = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
val_output_converted = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]

out = dice_acc(y_pred=val_output_converted, y=val_labels_converted)
acc, not_nans = dice_acc.aggregate()
print(out)
print(acc)
print(not_nans)