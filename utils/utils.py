from torch import sigmoid
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from functools import partial
from monai.networks.nets import SwinUNETR
from typing import Sequence

__all__ = ["post_pred_transform", "SwinInferer"]


def post_pred_transform(threshold: float = 0.5) -> callable:
    def _wrapper(x):
        x = sigmoid(x)
        x = AsDiscrete(threshold=threshold)(x)
        return x
    return _wrapper


def SwinInferer(model: SwinUNETR, roi_size: int | Sequence[int],
                sw_batch_size: int = 4, overlap: float = 0.5) -> sliding_window_inference:
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    return model_inferer
