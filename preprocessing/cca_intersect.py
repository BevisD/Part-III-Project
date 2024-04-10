import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from perlin_noise import PerlinNoise
from skimage import measure
import numpy as np
import torch


def create_blobs(threshold=0.0, rows=100, cols=100, seed=1):
    noise = PerlinNoise(octaves=10, seed=seed)

    img = np.array([[noise([i / cols, j / rows]) for j in range(cols)] for i in range(rows)])

    bg_mask = img < threshold
    fg_mask = img >= threshold
    img[bg_mask] = 0
    img[fg_mask] = 1

    return img


def create_blob(threshold=0.0, rows=100, cols=100, seed=1):
    img = create_blobs(threshold=threshold, rows=rows, cols=cols, seed=seed)
    labeled_image, num_labels = measure.label(img, return_num=True)

    max_i, max_vol = 0, 0
    for i in range(1, num_labels + 1):
        vol = np.count_nonzero(labeled_image == i)
        if vol > max_vol:
            max_vol = vol
            max_i = i

    return (labeled_image == max_i).astype(int)


def main():
    plt.rcParams['axes.facecolor'] = 'black'

    cmap_largest = ListedColormap(["red"])
    cmap_intersect = ListedColormap(["blue"])

    largest_lesion = create_blob(threshold=0.1, seed=2)
    lesions = create_blobs(threshold=0.1, seed=1)

    labeled_lesions = measure.label(lesions)

    intersecting = np.unique(labeled_lesions * largest_lesion)
    intersecting = intersecting[intersecting.nonzero()]
    intersecting_mask = np.isin(labeled_lesions, intersecting).astype(int)

    plt.imshow(largest_lesion, cmap=cmap_largest, alpha=0.5 * largest_lesion)
    plt.imshow(lesions, cmap="gray", alpha=0.5 * lesions)
    plt.imshow(intersecting_mask, cmap=cmap_intersect, alpha=0.5 * intersecting_mask)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.show()


def get_intersecting_components(label: torch.Tensor, pred: torch.Tensor):
    assert label.ndim == pred.ndim == 5
    assert label.shape[0] == pred.shape[0] == 1
    assert label.shape[1] == pred.shape[1] == 1

    device = pred.device

    label = label.squeeze().cpu()
    pred = pred.squeeze().cpu()

    omtm_pred_lesions = torch.Tensor(measure.label(pred == 1))
    pod_pred_lesions = torch.Tensor(measure.label(pred == 2))

    omtm_target_mask = (label == 1)
    pod_target_mask = (label == 2)

    omtm_intersect = torch.unique(omtm_target_mask * omtm_pred_lesions)
    omtm_intersect = omtm_intersect[omtm_intersect.nonzero()]

    pod_intersect = torch.unique(pod_target_mask * pod_pred_lesions)
    pod_intersect = pod_intersect[pod_intersect.nonzero()]

    omtm_mask = torch.isin(omtm_pred_lesions, omtm_intersect)
    pod_mask = torch.isin(pod_pred_lesions, pod_intersect)

    pred_mask = omtm_mask + pod_mask

    return (pred * pred_mask).unsqueeze(0).unsqueeze(0).to(device)


if __name__ == '__main__':
    # main()
    plt.rcParams['axes.facecolor'] = 'black'
    label_cmap = ListedColormap(["black", "red", "blue"])
    pred_cmap = ListedColormap(["black", "pink", "green"])

    omtm = create_blob(threshold=0.1, rows=200, cols=200, seed=1)
    pod = create_blob(threshold=0.1, rows=200, cols=200, seed=3) * 2
    label = omtm + pod

    pred = create_blobs(threshold=0.1, rows=200, cols=200, seed=8)
    pred = measure.label(pred) % 3

    pred = torch.Tensor(pred).view((1, 1, 1, *pred.shape))
    label = torch.Tensor(label).view((1, 1, 1, *label.shape))

    pred = get_intersecting_components(label, pred)

    pred = pred.squeeze()
    label = label.squeeze()

    plt.imshow(label, cmap=label_cmap, alpha=(label != 0)*0.5, vmin=0.0, vmax=2.0)
    plt.imshow(pred, cmap=pred_cmap, alpha=(pred != 0)*0.5, vmin=0.0, vmax=2.0)
    plt.show()
