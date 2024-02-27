import torch
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

IMG_SIZE = 96
IN_CHANNELS = 1
OUT_CHANNELS = 2
FEATURE_SIZE = 48
DROP_RATE = 0.2
ATTN_DROP_RATE = 0.0
DROPOUT_PATH_RATE = 0.0
USE_CHECKPOINT = True
SQUARED_PRED = False
PRETRAINED_PATH = "pretrained_models/swinunetr.pt"


def main() -> None:
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

if __name__ == '__main__':
    main()
