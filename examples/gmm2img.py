"""
GMM/RBF with equal weights and shared sigma representation of the given 3D model, which
is rotated and projected and then converted to 2D model image. A direct score should compare
this to the random cryo-EM image.

*Original code was in torch given by Julian, converted to jax.*
"""

import math
from functools import partial

import jax
import jax.numpy as jnp


def voxel_shifts(sigma_coord=2, k=3):
    """Create voxel shifts for a truncated Gaussian kernel"""
    radius = int(math.ceil(k * sigma_coord))
    coords = jnp.arange(-radius, radius + 1)
    x, y = jnp.meshgrid(coords, coords, indexing="ij")
    shifts = jnp.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=-1)
    norms = jnp.linalg.norm(shifts.astype(float), axis=-1)
    return shifts[norms <= radius]


def gauss_kernel(sigma_coord=2, k=3):
    """Create the truncated Gaussian kernel"""
    radius = int(math.ceil(k * sigma_coord))
    kernel_size = 2 * radius + 1
    coords = jnp.arange(-radius, radius + 1)
    x, y = jnp.meshgrid(coords, coords, indexing="ij")
    kernel = jnp.exp(-(x**2 + y**2) / (2 * sigma_coord**2))
    kernel = kernel / jnp.sum(kernel)
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
    return kernel


class GMM:
    def __init__(self, means, sigma, weight, image_size=136, offset=0):
        """
        Gaussian Mixture Model with equal weights and shared sigma
        to approximate the simulated Cryo-EM maps.

        Parameters:
        - means: Array of shape (N, 3), initial means of the Gaussians.
        - sigma: float, initial value for the shared sigma.
        - weight: float, initial value for the shared weight.
        - image_size: int, size of the output image.
        - offset: float, offset added to the output.
        """
        self.N = means.shape[0]  # Number of components
        self.D = means.shape[1]  # Dimensionality (should be 3)
        self.offset = offset

        # Initialize parameters
        self.means = means  # Shape: (N, 3)
        self.log_sigma = jnp.log(jnp.array(sigma))  # Log sigma for positivity
        self.log_weight = jnp.log(jnp.array(weight))  # Log weight for positivity

        self.image_size = image_size
        self.kernel = gauss_kernel(sigma_coord=sigma * image_size / 2)
        self.sigma_coord = jnp.exp(self.log_sigma) * self.image_size / 2
        self.voxel_shifts = voxel_shifts(sigma_coord=2, k=3)

    def update_params(self, means=None, log_sigma=None, log_weight=None):
        """Update model parameters"""
        if means is not None:
            self.means = means
        if log_sigma is not None:
            self.log_sigma = log_sigma
            self.sigma_coord = jnp.exp(self.log_sigma) * self.image_size / 2
        if log_weight is not None:
            self.log_weight = log_weight
        return self

    @partial(jax.jit, static_argnums=(0,))
    def _scatter_add(self, out, global_idx, contrib):
        """Perform scatter-add operation in JAX"""
        return out.at[global_idx].add(contrib)

    def images(self, rotations, scalar=1):
        """
        Generate 2D projection images from 3D points.

        Parameters:
        - rotations: Array of shape (B, 3, 3), batch of rotation matrices.
        - scalar: float, scaling factor for the output.

        Returns:
        - out: Array of shape (B, 1, image_size, image_size), generated images.
        - projected: Array of shape (B, N, 2), projected 2D points.
        """
        points = self.means
        grid_size = self.image_size
        offset = self.offset - 1
        sigma_coord = jnp.exp(self.log_sigma) * grid_size / 2
        norm_const = 2 * jnp.pi * sigma_coord**2
        B = rotations.shape[0]  # batch size
        N = points.shape[0]  # number of points
        shifts = self.voxel_shifts
        out = jnp.full((B * grid_size**2,), offset, dtype=jnp.float32)
        strides = jnp.array([1, grid_size], dtype=jnp.int32).reshape(1, 1, 2)

        # batch rotation and projection onto the xy-plane
        projected = jnp.einsum("nj,bij->bni", points, rotations)[..., :2]

        # origin is the center of the image
        origin = jnp.ones((B, 1, 2))

        # Compute continuous grid coordinates
        # The grid is assumed to cover [-1, 1] in each dimension
        # The scaling factor converts from world coordinates to grid coordinates
        coord = (projected + origin) * (grid_size - 1) / 2.0  # (B, N, 2)

        # Compute the reference voxel coordinates
        ref_voxel = jnp.floor(coord).astype(jnp.int32)
        ref_voxel = jnp.clip(ref_voxel, 0, grid_size - 1)

        # Compute the deltas between reference and coordinates
        deltas_lower = coord - ref_voxel  # (B, N, 2)

        # Process each shift separately and accumulate results
        def process_shift(out, shift_idx):
            s = shifts[shift_idx]  # shape (2,)
            s_b = jnp.broadcast_to(s.reshape(1, 1, 2), (B, N, 2))

            # The voxel index in each dimension is the reference voxel plus the shift
            voxel_idx = ref_voxel + s_b  # (B, N, 2)

            # Check that each coordinate is within bounds
            valid_x = (voxel_idx[..., 0] < grid_size) & (voxel_idx[..., 0] >= 0)
            valid_y = (voxel_idx[..., 1] < grid_size) & (voxel_idx[..., 1] >= 0)
            valid = valid_x & valid_y

            # Compute weight factors for valid points
            weight_factor = (
                jnp.exp(
                    -(jnp.sum((-deltas_lower + s_b) ** 2, axis=-1))
                    / (2 * sigma_coord**2)
                )
                / norm_const
            )

            # Weigh the contribution
            total_contrib = scalar * jnp.exp(self.log_weight) * weight_factor  # (B, N)

            # Zero out contributions for points whose indices fall outside the grid
            total_contrib = total_contrib * valid

            # Convert the multi-dimensional voxel indices to a flat index
            flat_idx = jnp.sum(
                (voxel_idx * jnp.expand_dims(valid, -1)) * strides, axis=-1
            )
            batch_offsets = jnp.arange(B).reshape(B, 1) * grid_size**2
            global_idx = flat_idx + batch_offsets  # (B, N)

            # Scatter-add the contributions into the output
            out = self._scatter_add(out, global_idx.flatten(), total_contrib.flatten())
            return out

        # Process all shifts and accumulate results
        for i in range(len(shifts)):
            out = process_shift(out, i)

        out = out.reshape(B, 1, grid_size, grid_size)
        return out, projected


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    # Initialize random number generator with a seed for reproducibility
    key = jax.random.PRNGKey(0)

    # Generate random means
    means = jax.random.uniform(key, (200000, 3)) - 0.5
    sigma = 0.1
    weight = 1.0

    # Generate random rotations
    rotations = jnp.array([R.random().as_matrix() for _ in range(4)], dtype=jnp.float32)

    # Create GMM instance
    gmm = GMM(means, sigma, weight, image_size=136)

    # Generate images and projected points
    images, points2d = gmm.images(rotations)

    # Plot the results
    num_images = images.shape[0]
    cols = 2
    rows = (num_images + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axs = [axs]  # Make axs indexable if there's only one row

    for i in range(rows * cols):
        if i < num_images:
            ax = axs[i // cols] if cols == 1 else axs[i // cols, i % cols]
            ax.imshow(images[i, 0], cmap="gray")
            ax.axis("off")
        else:
            if cols == 1:
                axs[i // cols].remove()
            else:
                axs[i // cols, i % cols].remove()

    plt.tight_layout()
    plt.show()
