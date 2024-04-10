import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from perlin_noise import PerlinNoise
from skimage import measure
import numpy as np


def create_blobs(threshold=0.0, rows=100, cols=100, seed=1):
    noise = PerlinNoise(octaves=10, seed=seed)

    img = np.array([[noise([i / cols, j / rows]) for j in range(cols)] for i in range(rows)])

    bg_mask = img < threshold
    fg_mask = img >= threshold
    img[bg_mask] = 0
    img[fg_mask] = 1

    return img


def show_connectivities():
    img = create_blobs(threshold=0.102, rows=200, cols=200, seed=1)
    labeled_image_1, num_labels_1 = measure.label(img, connectivity=1, return_num=True)
    labeled_image_2, num_labels_2 = measure.label(img, connectivity=2, return_num=True)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].set_title("Segmentation Mask")
    axs[0].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

    axs[1].set_title(f"No Diagonals: # = {num_labels_1}")
    axs[1].imshow(labeled_image_1, cmap="tab20", alpha=1.0 * (labeled_image_1 != 0))
    axs[1].set_facecolor("black")

    axs[2].set_title(f"Diagonals: # = {num_labels_2}")
    axs[2].imshow(labeled_image_2, cmap="tab20", alpha=1.0 * (labeled_image_2 != 0))
    axs[2].set_facecolor("black")

    rectangle_1 = Rectangle((1, 70), 43, 50, linewidth=2, edgecolor="red", fill=False)
    rectangle_2 = Rectangle((1, 70), 43, 50, linewidth=2, edgecolor="red", fill=False)
    axs[1].add_patch(rectangle_1)
    axs[2].add_patch(rectangle_2)

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def show_largest():
    cmap = ListedColormap(["red"])

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    for row in range(3):
        for col in range(3):
            img = create_blobs(threshold=0.102, rows=200, cols=200, seed=col + 3 * row)
            labeled_image, num_labels = measure.label(img, connectivity=1, return_num=True)

            max_i, max_vol = 0, 0
            for i in range(1, num_labels+1):
                vol = np.count_nonzero(labeled_image == i)
                if vol > max_vol:
                    max_vol = vol
                    max_i = i

            axs[row, col].imshow(img, cmap="gray")
            axs[row, col].imshow(labeled_image == max_i, cmap=cmap, alpha=1.0*(labeled_image == max_i))
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])

    plt.tight_layout()
    plt.show()

# show_connectivities()
show_largest()
