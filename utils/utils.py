from torch import sigmoid, nn
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from functools import partial
from monai.networks.nets import SwinUNETR
from typing import Sequence

__all__ = ["post_pred_transform",
           "SwinInferer",
           "response_accuracy"]


def post_pred_transform(threshold: float = 0.5) -> callable:
    def _wrapper(x):
        x = sigmoid(x)
        x = AsDiscrete(threshold=threshold)(x)
        return x
    return _wrapper


def response_accuracy(y_pred, y):
    assert y.ndim == 1 and y.size() == y_pred.size()
    y_pred = y_pred > 0.5
    return (y == y_pred).sum().item() / y.size(0)


def SwinInferer(model: nn.Module, roi_size: int | Sequence[int],
                sw_batch_size: int = 4, overlap: float = 0.5) -> sliding_window_inference:
    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    return model_inferer
