# Uses 2D projection images from cryoEM

"""
Benchmarks the speed of kernel correlation vs Gaussian mixture (soft ICP?) methods for a simple 2D-2D registration.
"""

# %%
import pathlib

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np

from bayalign.pointcloud import PointCloud
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians, Registration
from bayalign.utils import create_rotation, take_time


def extract_index(filename):
    """Extract image index from filename."""
    *a, index = str(filename).partition("_image")
    return int(index.split("_")[0])


def load_model2d():
    """Load 2D point cloud models precomputed with EM representing class
    averages of the 80S ribosome."""

    path = pathlib.Path("data/ribosome_80S/model2d_em")
    files = sorted(path.glob("ribosome*.npz"), key=extract_index)
    clouds = []
    sigmas = []

    for filename in files:
        data = np.load(str(filename))
        clouds.append(data["points"] - data["points"].mean(0))
        sigmas.append(float(data["sigma"]))

    return np.array(clouds), np.array(sigmas)


def cost(
    reg_score: Registration,
    angles,
):
    """
    Calculates kernel correlation cost over angular grid

    :param kernel_correlation: instance
                instance of KernelCorrelation
    :param rotation_matrix: instance
                instance of RotationMatrix
    :param angles: array_like
                angular grid
    :return: list
                list of calculated cost over angular grid
    """
    corrs = []
    for angle in angles:
        rotmats = create_rotation(angle)
        corrs.append(reg_score.cost(rotmats))

    return corrs


def make_subplots(nrows, ncols):
    subplot_kw = dict(xticks=[], yticks=[], aspect=1.0)

    if nrows >= ncols:
        scale = int(nrows / ncols) + 1
    else:
        scale = int(ncols / nrows) + 1

    fig, subplots = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * scale, nrows * scale),
        sharex="all",
        sharey="all",
        subplot_kw=subplot_kw,
    )
    return fig, subplots


def show_pc(clouds, nrows=5, ncols=10):
    """Show point clouds as a scatter plot."""

    scatter_kw = dict(alpha=0.3, color="k", s=10)
    fig, subplots = make_subplots(nrows, ncols)

    for subplot, cloud in zip(subplots.flat, clouds):
        subplot.scatter(*cloud.T, **scatter_kw)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    clouds, sigmas = load_model2d()

    i, j = 0, 5
    target = PointCloud(clouds[i], weights=None)
    source = PointCloud(clouds[j], weights=None)

    sigma = (sigmas[i] ** 2 + sigmas[j] ** 2) ** 0.5

    # systematic angular grid for evaluating kernel correlation
    n_angles = 1000
    angles = jnp.linspace(0.0, 2 * jnp.pi, n_angles, endpoint=False)

    # convert every angle to 2D rotation matrix
    rotmats = jnp.asarray([create_rotation(angle) for angle in angles])

    # %%
    # instantiate kernel correlation
    score_kc = KernelCorrelation(target, source, sigma)

    # instantiate mixture spherical gaussians
    score_msg = MixtureSphericalGaussians(target, source, sigma)

    # cost is negative log prob
    with take_time(f"evaluating KDTreeKernelCorrelation {n_angles} times"):
        kc_logprobs = np.array([score_kc.log_prob(rotmat) for rotmat in rotmats])

    with take_time(f"evaluating MixtureSphericalGaussians {n_angles} times"):
        msg_logprobs = np.array(cost(score_msg, angles))

    # %%
    # transform the source for the least cost
    rotation = create_rotation(angles[np.argmax(kc_logprobs)])
    source_transformed = source.transform_positions(rotation)

    # plotting
    fig = show_pc(
        [target.positions, source.positions, source_transformed],
        nrows=1,
        ncols=3,
    )
    fig.axes[0].set_title("target")
    fig.axes[1].set_title("source")
    fig.axes[2].set_title("rotated source")
    plt.show()

# %%
