from anisoswinunetr import AnisoSwinUNETR
from monai.networks.nets import SwinUNETR
import torch

torch.random.manual_seed(0)
model_1 = AnisoSwinUNETR(
    img_size=(32, 256, 256),
    in_channels=1,
    out_channels=3,
    feature_size=24,
    patch_size=(2, 2, 2),
    window_size=(7, 7, 7)
)

torch.random.manual_seed(0)
model_2 = SwinUNETR(
    img_size=(32, 256, 256),
    in_channels=1,
    out_channels=3,
    feature_size=24,
)

# for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
#     if p1.shape != p2.shape:
#         print(f"{p1.shape}\t{p2.shape}")


pytorch_total_params = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
print("Total model_1 parameters count", pytorch_total_params)

pytorch_total_params = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
print("Total model_2 parameters count", pytorch_total_params)