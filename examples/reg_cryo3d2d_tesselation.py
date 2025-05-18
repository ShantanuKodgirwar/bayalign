# 3D-to-2D Rigid registration by tesselation of the unit sphere for systematic
# rotations (generate ground truth). This the done as follows:
# 1. Load class averages of the 80S ribosome and convert to a dense point cloud
# 2. Fit a smaller 2D point cloud to this using expectation maximization which is our target
# 3. Tesselate the unit sphere to generate a set of rotations as quaternions
# 4. Project the 3D class average onto the 2D model and find the quaternion that best aligns
# with the projection either via a score of kernel correlation or mixture of spherical gaussians.

# %%
import os

import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np

from bayalign.cryo_utils import (
    fit_model2d,
    load_class_average,
    pointcloud_from_class_avg,
)
from bayalign.pointcloud import RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians
from bayalign.sphere_tesselation import tessellate_rotations
from bayalign.utils import take_time


def run_scoring_method(
    scoring_metric,
    prob,
    quaternions,
    target_fit2d,
    class_avg_idx,
    k_neighbours,
    save_path,
    load_precomputed=False,
):
    """Run the specified scoring method and return the best quaternion and logprobs."""
    if scoring_metric == "KC":
        savename = f"KC_3d2d_C600_K{k_neighbours}_projection_idx{class_avg_idx}.npz"
        title = "Kernel Correlation"
        log_message = "kernel correlations"
    else:  # "MSG"
        savename = f"MSG_3d2d_C600_K{k_neighbours}_projection_idx{class_avg_idx}.npz"
        title = "Mixture of Spherical Gaussians"
        log_message = "log prob. mixture of spherical gaussians"

    savepath = os.path.join(save_path, savename)

    if os.path.exists(savepath) and load_precomputed:
        print(f"loading precomputed {log_message}")
        logprobs = jnp.load(savepath)["log_prob"]
    else:
        print(f"evaluating {log_message} ...")
        with take_time(f"evaluating {scoring_metric} {len(quaternions)} times"):
            logprobs = jnp.asarray([prob.log_prob(q) for q in quaternions])
            jnp.savez(
                savepath,
                log_prob=logprobs,
            )

    # best quaternion
    q_best = quaternions[jnp.argmax(logprobs)]

    # Create histogram plot
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.hist(logprobs, bins=100, color="k", density=True, alpha=0.4)
    ax.axvline(prob.log_prob(q_best), color="r", ls="--", label="optimum")
    ax.set_xlabel("log probability" if scoring_metric == "MSG" else "log prob.")
    ax.legend()
    ax.semilogy()
    fig.tight_layout()

    return q_best, logprobs


def plot_results(
    image,
    target,
    target_fit2d,
    source,
    q_best,
    save_path,
    scoring_metric,
    k_neighbours,
    class_avg_idx,
    savefig=True,
):
    """Plot the results of the registration."""
    limit = len(image) // 2 * 2.68

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex="all", sharey="all")
    ax1, ax2, ax3, ax4 = list(axes.flat)
    for ax in list(axes.flat):
        ax.set_xlim(-0.7 * limit, 0.7 * limit)
        ax.set_ylim(-0.7 * limit, 0.7 * limit)
        ax.set_axis_off()

    ax1.set_title("Experimental Projection Image")
    ax1.imshow(image.T, extent=(-limit, limit, -limit, limit), origin="lower")

    ax2.set_title("Segmented Target: 2D point-cloud")
    ax2.scatter(*target.positions.T, c=target.weights, alpha=0.2, s=20)

    ax3.set_title("Ground Truth: 2D Model")
    ax3.imshow(image.T, extent=(-limit, limit, -limit, limit), origin="lower")
    ax3.scatter(*target_fit2d.positions.T, color="w", alpha=0.2, s=20)

    ax4.set_title("Source: Rotated and projected 3D Model")
    ax4.imshow(image.T, extent=(-limit, limit, -limit, limit), origin="lower")
    ax4.scatter(*source.transform_positions(q_best).T, color="w", alpha=0.15, s=20)

    for ax in (ax2, ax3, ax4):
        ax.set_aspect(1 / ax.get_data_ratio())

    if savefig:
        savename = f"reg_{scoring_metric}_3dto2d_C600_K{k_neighbours}_projection_idx{class_avg_idx}.png"
        fig.savefig(
            os.path.join(save_path, savename),
            dpi=150,
        )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # directory to save results
    save_path = "results/reg_3d2d_tesselation_KC/"
    os.makedirs(save_path, exist_ok=True)

    # scoring metric ("MSG" or "KC")
    scoring_metric = "KC"  # default is KC

    # load target; index specifies which class avg will be used as a target
    class_avg_idx = 10
    pixelsize = 2.68  # Angstrom
    image = load_class_average(class_avg_idx)
    target = pointcloud_from_class_avg(
        image,
        pixelsize=pixelsize,
    )

    # no of points/ RBF kernels for source
    n_particles = 2000

    # load 3d model
    model = np.load(f"data/ribosome_80S/model3d_{n_particles}.npz")

    # instance of the 3D point cloud source with a correction of center of mass
    source = RotationProjection(model["positions"], model["weights"]).centered_copy()

    # fit target with (unweighted) 2D point cloud using EM
    target_fit2d, sigma_truth = fit_model2d(target, n_particles, n_iter=100, k=50)
    k_neighbours = 20

    # generate quaternions (600-cell) for systematic rotation space search
    n_discretize = 1
    quaternions = jnp.asarray(tessellate_rotations(n_discretize))
    print(f"number of quaternions: {len(quaternions)}")

    # Select the scoring approach
    if scoring_metric == "KC":
        prob = KernelCorrelation(
            target_fit2d,
            source,
            sigma_truth,
            k=k_neighbours,
            beta=20.0,
        )
    elif scoring_metric == "MSG":  # "MSG"
        prob = MixtureSphericalGaussians(
            target_fit2d,
            source,
            sigma_truth,
            k=k_neighbours,
            beta=1.0,
        )
    else:
        raise ValueError(f"Unknown scoring metric: {scoring_metric}")

    # Parameters for processing
    load_precomputed = True
    savefig = True

    # Run the selected scoring method
    q_best, logprobs = run_scoring_method(
        scoring_metric,
        prob,
        quaternions,
        target_fit2d,
        class_avg_idx,
        k_neighbours,
        save_path,
        load_precomputed,
    )

    # Plot results
    fig = plot_results(
        image,
        target,
        target_fit2d,
        source,
        q_best,
        save_path,
        scoring_metric,
        k_neighbours,
        class_avg_idx,
        savefig,
    )

    plt.show()
