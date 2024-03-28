import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def plot_box(coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    faces = []
    face_indices = [[0, 1, 2, 3, 0],
                    [4, 5, 6, 7, 7],
                    [0, 3, 7, 4, 0],
                    [1, 2, 6, 5, 1],
                    [0, 1, 5, 4, 0],
                    [2, 3, 7, 6, 2]]

    for f, _ in enumerate(face_indices):
        face = np.zeros([5, 3])
        for i in range(5):
            face[i, :] = coords[face_indices[f][i]]

        faces.append(face)

    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1, 1, 1))

    ax.axes.set_xlim3d(left=-50, right=50)
    ax.axes.set_ylim3d(bottom=-312.5, top=312.5)
    ax.axes.set_zlim3d(bottom=-312.5, top=312.5)

    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axes.set_zticks([])

    plt.tight_layout()
    plt.show()


def main():
    H, W, D = 32, 256, 256
    coords = 0.5 * np.array([
        [-H, -W, -D],
        [-H, -W, D],
        [-H, W, D],
        [-H, W, -D],
        [H, -W, -D],
        [H, -W, D],
        [H, W, D],
        [H, W, -D],
    ])

    theta_ranges = [
        (-0.1, 0.1),  # XY
        (-0.1, 0.1),  # XZ
        (-1.5, 1.5)]  # YZ
    axes = [(0, 1),  # XY
            (0, 2),  # XZ
            (1, 2)]  # YZ

    shears = [(-0.15, +0.15),  # XY
              (-0.10, +0.10),  # YZ
              (-0.10, +0.10)]  # XZ

    scales = [(0.8, 1.1),
              (0.8, 1.1),
              (1.0, 1.0)]

    flips = [0.5, 0.5, 0.5]

    affine = identity_affine(dim=3, N=1)

    flip_affine = rand_flip_affine(probs=flips, dim=3, N=1)
    affine = multiply_affines(flip_affine, affine)

    scale_affine = rand_scale_affine(dim=3, scales=scales, N=1)
    affine = multiply_affines(scale_affine, affine)

    shear_affine = rand_shear_affine(dim=3, shears=shears, N=1)
    affine = multiply_affines(shear_affine, affine)

    for theta_range, axes in zip(theta_ranges, axes):
        theta_affine = rand_rotate_affine(dim=3, theta_range=theta_range, axes=axes)
        affine = multiply_affines(theta_affine, affine)

    # print(affine)

    affine = affine[0, :, :3]
    new_coords = np.matmul(affine, coords.T).T
    plot_box(new_coords)


if __name__ == '__main__':
    from affine import rand_rotate_affine, rand_flip_affine, rand_shear_affine, rand_scale_affine, identity_affine, \
        multiply_affines

    main()
