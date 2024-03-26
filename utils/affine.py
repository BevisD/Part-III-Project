from typing import Sequence

import torch
import torch.nn.functional as F
import monai.transforms as transforms


def rand_uniform(low: float, high: float, N=1):
    return low + torch.rand(N) * (high - low)


def identity_affine(dim: int = 3, N: int = 1):
    return F.pad(torch.diag(torch.ones(dim)), (0, 1)).unsqueeze(0).repeat(N, 1, 1)


def multiply_affines(left, right):
    left = F.pad(left, (0, 0, 0, 1))
    right = F.pad(right, (0, 0, 0, 1))
    left[..., -1, -1] = 1

    affine = torch.einsum('bij,bjk->bik', left, right)
    return affine[..., :-1, :]


def random_pos_neg(prob: float = 0.5, N: int = 1):
    # Probability of returning negative values
    return 2 * (torch.rand(N) > prob) - 1


def rand_flip_affine(probs: Sequence[float] = (0.5, 0.5, 0.5), dim: int = 3, N: int = 1):
    affine = identity_affine(dim=dim, N=N)
    for i, p in enumerate(probs):
        affine[..., i, i] = random_pos_neg(p, N=N)
    return affine


def rand_scale_affine(scales: Sequence[Sequence[float]], dim: int = 3, N: int = 1):
    affine = identity_affine(dim=dim, N=N)
    for i, s in enumerate(scales):
        affine[..., i, i] = rand_uniform(*s, N=N)
    return affine


def rand_shear_affine(shears: Sequence[Sequence[float]], dim: int = 3, N: int = 1):
    affine = identity_affine(dim=dim, N=N)
    for i, shear_range in enumerate(shears):
        shear_val = rand_uniform(*shear_range, N=N)
        affine[..., i, (i + 1) % dim] = shear_val
        affine[..., (i + 1) % dim, i] = shear_val
    return affine


def rand_rotate_affine(theta_range: Sequence[float],
                       axes: Sequence[float], dim=3, N: int = 1):
    thetas = rand_uniform(*theta_range, N=N)

    cos_thetas = torch.cos(thetas)
    sin_thetas = torch.sin(thetas)

    affine = identity_affine(dim=dim, N=N)
    affine[..., axes[0], axes[0]] = +cos_thetas
    affine[..., axes[0], axes[1]] = -sin_thetas
    affine[..., axes[1], axes[0]] = +sin_thetas
    affine[..., axes[1], axes[1]] = +cos_thetas
    return affine


class RandAffineTransformd(transforms.MapTransform):
    def __init__(self,
                 keys: Sequence[str] | str,
                 mode: Sequence[str] | str,
                 flips: Sequence[float] = None,
                 thetas: Sequence[Sequence[float]] = None,
                 axes: Sequence[Sequence[float]] = None,
                 scales: Sequence[Sequence[float]] = None,
                 shears: Sequence[Sequence[float]] = None,

                 allow_missing_keys=False) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.flips = flips
        self.thetas = thetas
        self.scales = scales
        self.shears = shears
        self.mode = [mode] * len(self.keys) if isinstance(mode, str) else mode
        self.axes = axes
        self.device, self.type = ("cuda", torch.half) if torch.cuda.is_available() else ("cpu", torch.float32)

    def gen_rand_affine(self, N: int = 1):
        affine = identity_affine(dim=3, N=N)

        # Flip Image
        if self.flips is not None:
            flip_affine = rand_flip_affine(dim=3, probs=self.flips, N=N)
            affine = multiply_affines(flip_affine, affine)

        # Scale Image
        if self.scales is not None:
            scale_affine = rand_scale_affine(dim=3, scales=self.scales, N=N)
            affine = multiply_affines(scale_affine, affine)

        # Shear Image
        if self.shears is not None:
            shear_affine = rand_shear_affine(dim=3, shears=self.shears, N=N)
            affine = multiply_affines(shear_affine, affine)

        # Rotate Image
        if self.thetas is not None:
            for theta_range, axes in zip(self.thetas, self.axes):
                theta_affine = rand_rotate_affine(dim=3, theta_range=theta_range, axes=axes, N=N)
                affine = multiply_affines(theta_affine, affine)

        return affine

    def __call__(self, batch: dict) -> dict:
        size = batch[self.keys[0]].shape
        N = size[0]
        affine = self.gen_rand_affine(N=N).to(self.type).to(self.device)
        grid = F.affine_grid(affine, size, align_corners=True)

        for key, mode in zip(self.keys, self.mode):
            if key in batch:
                batch[key] = batch[key].to(self.type).to(self.device)
                batch[key] = F.grid_sample(batch[key],
                                           grid,
                                           align_corners=True,
                                           mode=mode)
            else:
                raise ValueError(f"Key '{key}' is not in data")
        return batch

