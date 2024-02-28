import monai.transforms
from torch import argmax
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from functools import partial
from monai.networks.nets import SwinUNETR

__all__ = ["post_pred_transform", "SwinInferer"]


def post_pred_transform(num_classes: int, dim=1) -> callable:
    def _wrapper(x):
        return AsDiscrete(to_onehot=num_classes)(argmax(x, dim=dim))
    return _wrapper


def SwinInferer(model: SwinUNETR, roi_size: int,
                sw_batch_size: int = 4, overlap: int = 0.5) -> sliding_window_inference:
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    return model_inferer
