import os

import matplotlib.pyplot as plt
import numpy as np
from jax import random
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from bayalign.inference import (
    MetropolisHastings,
    RejectionSphericalSliceSampler,
    ShrinkageSphericalSliceSampler,
    SphericalHMC,
)
from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import GaussianMixtureModel, KernelCorrelation
from bayalign.sphere_utils import distance, sample_sphere
from bayalign.utils import take_time


def create_synthetic_pointcloud(n_points=512, seed=42):
    """
    Create a 3D point cloud with distinct geometric features.
    """
    np.random.seed(seed)
    points = []

    # L-bracket structure (200 points)
    n_vertical = 100
    t_vert = np.linspace(0, 2, n_vertical)
    x_vert = np.zeros(n_vertical)
    y_vert = np.zeros(n_vertical)
    z_vert = t_vert

    n_horizontal = 100
    t_horiz = np.linspace(0, 2, n_horizontal)
    x_horiz = t_horiz
    y_horiz = np.zeros(n_horizontal)
    z_horiz = np.zeros(n_horizontal)

    l_points = np.column_stack(
        [
            np.concatenate([x_vert, x_horiz]),
            np.concatenate([y_vert, y_horiz]),
            np.concatenate([z_vert, z_horiz]),
        ]
    )
    points.append(l_points)

    # Helical structure (200 points)
    n_helix = 200
    t_helix = np.linspace(0, 4 * np.pi, n_helix)
    radius = 0.5
    x_helix = radius * np.cos(t_helix) + 1
    y_helix = radius * np.sin(t_helix) + 1
    z_helix = t_helix / (2 * np.pi) + 0.5

    helix_points = np.column_stack([x_helix, y_helix, z_helix])
    points.append(helix_points)

    # Corner feature points
    remaining = n_points - 400
    corner_points = np.array(
        [
            [0, 0, 0],
            [0, 0, 0.1],
            [0, 0.1, 0],
            [0.1, 0, 0],
            [0, 0.1, 0.1],
            [0.1, 0, 0.1],
            [0.1, 0.1, 0],
        ]
    )

    # Add random points for robustness
    if remaining > 7:
        noise_scale = 0.05
        n_l_noise = (remaining - 7) // 2
        l_noise = l_points[
            np.random.choice(len(l_points), n_l_noise)
        ] + np.random.normal(0, noise_scale, (n_l_noise, 3))

        n_h_noise = remaining - 7 - n_l_noise
        h_noise = helix_points[
            np.random.choice(len(helix_points), n_h_noise)
        ] + np.random.normal(0, noise_scale, (n_h_noise, 3))

        points.extend([l_noise, h_noise])

    # Combine all points
    all_points = np.vstack([corner_points] + points)

    # Ensure exactly n_points
    if len(all_points) > n_points:
        indices = np.random.choice(len(all_points), n_points, replace=False)
        all_points = all_points[indices]
    elif len(all_points) < n_points:
        deficit = n_points - len(all_points)
        extra_indices = np.random.choice(len(all_points), deficit, replace=True)
        extra_points = all_points[extra_indices] + np.random.normal(
            0, 0.01, (deficit, 3)
        )
        all_points = np.vstack([all_points, extra_points])

    return all_points


def plot_registration_results(
    target, source, transformed_source, title_prefix="Registration Results"
):
    """
    Plot three panels showing the registration process:
    1. 2D target point cloud
    2. 3D source point cloud
    3. 2D target overlaid with transformed source

    Args:
        target: PointCloud object with 2D positions
        source: RotationProjection object with 3D points
        transformed_source: 2D numpy array of transformed source points
        title_prefix: Prefix for the overall title
    """
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: 2D Target
    ax1 = fig.add_subplot(131)
    ax1.scatter(
        target.positions[:, 0],
        target.positions[:, 1],
        c="crimson",
        alpha=0.7,
        s=25,
        edgecolors="darkred",
        linewidth=0.5,
    )
    ax1.set_title("2D Target Point Cloud")
    ax1.set_xlabel("X'")
    ax1.set_ylabel("Y'")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Panel 2: 3D Source
    ax2 = fig.add_subplot(132, projection="3d")
    source_points = source.positions  # Get the 3D points from RotationProjection
    ax2.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        c="steelblue",
        alpha=0.7,
        s=20,
    )
    ax2.set_title("3D Source Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.grid(True, alpha=0.3)

    # Set equal aspect ratio for 3D plot
    max_range = (
        np.array(
            [
                source_points[:, 0].max() - source_points[:, 0].min(),
                source_points[:, 1].max() - source_points[:, 1].min(),
                source_points[:, 2].max() - source_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (source_points[:, 0].max() + source_points[:, 0].min()) * 0.5
    mid_y = (source_points[:, 1].max() + source_points[:, 1].min()) * 0.5
    mid_z = (source_points[:, 2].max() + source_points[:, 2].min()) * 0.5

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    # Panel 3: Target + Transformed Source Overlay
    ax3 = fig.add_subplot(133)
    # Plot target first (background)
    ax3.scatter(
        target.positions[:, 0],
        target.positions[:, 1],
        c="crimson",
        alpha=0.5,
        s=20,
        label="Target 2D",
        edgecolors="darkred",
        linewidth=0.3,
    )
    # Plot transformed source on top
    ax3.scatter(
        transformed_source[:, 0],
        transformed_source[:, 1],
        c="steelblue",
        alpha=0.7,
        s=20,
        label="Transformed Source",
        edgecolors="darkblue",
        linewidth=0.3,
    )
    ax3.set_title("Target + Transformed Source Overlay")
    ax3.set_xlabel("X'")
    ax3.set_ylabel("Y'")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_aspect("equal", adjustable="box")

    # Set overall title
    # fig.suptitle(title_prefix, fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle
    plt.show()

    return fig


def plot_registration_with_metrics(
    target,
    source,
    transformed_source,
    q_best,
    true_rotation,
    scoring_metric="Unknown",
    sampler="Unknown",
):
    """
    Enhanced version with metrics and quaternion information.
    """
    fig = plt.figure(figsize=(20, 7))

    # Panel 1: 2D Target
    ax1 = fig.add_subplot(141)
    ax1.scatter(
        target.positions[:, 0],
        target.positions[:, 1],
        c="crimson",
        alpha=0.7,
        s=25,
        edgecolors="darkred",
        linewidth=0.5,
    )
    ax1.set_title("2D Target Point Cloud")
    ax1.set_xlabel("X'")
    ax1.set_ylabel("Y'")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Panel 2: 3D Source
    ax2 = fig.add_subplot(142, projection="3d")
    source_points = source.positions
    ax2.scatter(
        source_points[:, 0],
        source_points[:, 1],
        source_points[:, 2],
        c="steelblue",
        alpha=0.7,
        s=20,
    )
    ax2.set_title("3D Source Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.grid(True, alpha=0.3)

    # Set equal aspect ratio for 3D plot
    max_range = (
        np.array(
            [
                source_points[:, 0].max() - source_points[:, 0].min(),
                source_points[:, 1].max() - source_points[:, 1].min(),
                source_points[:, 2].max() - source_points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (source_points[:, 0].max() + source_points[:, 0].min()) * 0.5
    mid_y = (source_points[:, 1].max() + source_points[:, 1].min()) * 0.5
    mid_z = (source_points[:, 2].max() + source_points[:, 2].min()) * 0.5

    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    # Panel 3: Target + Transformed Source Overlay
    ax3 = fig.add_subplot(143)
    ax3.scatter(
        target.positions[:, 0],
        target.positions[:, 1],
        c="crimson",
        alpha=0.5,
        s=20,
        label="Target 2D",
        edgecolors="darkred",
        linewidth=0.3,
    )
    ax3.scatter(
        transformed_source[:, 0],
        transformed_source[:, 1],
        c="steelblue",
        alpha=0.7,
        s=20,
        label="Transformed Source",
        edgecolors="darkblue",
        linewidth=0.3,
    )
    ax3.set_title("Registration Result")
    ax3.set_xlabel("X'")
    ax3.set_ylabel("Y'")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_aspect("equal", adjustable="box")

    # Panel 4: Metrics and Information
    ax4 = fig.add_subplot(144)
    ax4.axis("off")

    # Calculate registration error (simple MSE between closest points)

    distances = cdist(transformed_source, target.positions)
    min_distances = np.min(distances, axis=1)
    registration_error = np.mean(min_distances)

    # Relative rotation error
    angle_error_rad = distance(true_rotation, q_best)

    metrics_text = f"""Registration Metrics:

    Scoring Method: {scoring_metric}
    Sampler: {sampler}

    Ground Truth Quaternion:
    [{true_rotation[0]:.4f}, {true_rotation[1]:.4f}, 
    {true_rotation[2]:.4f}, {true_rotation[3]:.4f}]

    Best Predicted Quaternion:
    [{q_best[0]:.4f}, {q_best[1]:.4f}, 
    {q_best[2]:.4f}, {q_best[3]:.4f}]

    Registration Error:
    Mean Distance: {registration_error:.4f}
    
    Rotation Error:
    Angular Error: {angle_error_rad:.2f} rad

    Point Cloud Stats:
    Target Points: {len(target.positions)}
    Source Points: {len(source_points)}
    """

    ax4.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    fig.suptitle(
        f"Registration Results: {scoring_metric} with {sampler}", fontsize=16, y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    return fig, registration_error, angle_error_rad


def main():
    # scoring method (Kernel correlation or Mixture model (Gaussian))
    SCORING_METRIC = ["KC", "GMM"][1]
    SAMPLER = ["sss-reject", "sss-shrink", "hmc", "mh"][1]
    TARGET_SEED = [769, 5726, 2871][0]

    # sampler params
    n_samples = 100
    burnin = 0.2
    load_precomputed = False

    # Create the point cloud
    points = create_synthetic_pointcloud(512)

    print(f"Point cloud shape: {points.shape}")
    print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    source = RotationProjection(points)
    # Create a random rotation for ground truth
    true_rotation = R.random(random_state=TARGET_SEED).as_quat()
    print("Ground truth quaternion:")
    print(true_rotation)

    # Create target by rotating source and adding some noise
    pointcloud = source.transform_positions(true_rotation)
    target = PointCloud(pointcloud)

    if SCORING_METRIC == "GMM":
        print("Creating Gaussian Mixture Model target pdf")
        target_pdf = GaussianMixtureModel(
            target,
            source,
            sigma=1.0,
            k=20,
            beta=1.0,
        )
    elif SCORING_METRIC == "KC":
        print("Creating Kernel Correlation target pdf")
        target_pdf = KernelCorrelation(
            target,
            source,
            sigma=1.0,
            k=20,
            beta=20.0,
        )
    else:
        raise ValueError(f"Unknown scoring metric: {SCORING_METRIC}")

    # Select a sampler
    key = random.key(645)
    init_sampler_state = sample_sphere(key, d=3)
    if SAMPLER == "sss-reject":
        sampler = RejectionSphericalSliceSampler(
            target_pdf,
            init_sampler_state,
            seed=123,
        )
    elif SAMPLER == "sss-shrink":
        sampler = ShrinkageSphericalSliceSampler(
            target_pdf,
            init_sampler_state,
            seed=123,
        )
    elif SAMPLER == "hmc":
        sampler = SphericalHMC(
            target_pdf,
            init_sampler_state,
            seed=123,
        )
    elif SAMPLER == "mh":
        sampler = MetropolisHastings(
            target_pdf,
            init_sampler_state,
            seed=123,
        )
    else:
        pass

    # Sample from the target distribution and compute log probabilities
    filepath = (
        f"results/reg_3d2d_synthetic/{SCORING_METRIC}_{SAMPLER}_seed_{TARGET_SEED}.npz"
    )

    if load_precomputed and os.path.isfile(filepath):
        print(f"Loading precomputed samples from {filepath}")
        samples_q = np.load(filepath)["samples"]
        log_probs = np.load(filepath)["log_probs"]
    else:
        # sample the rotations
        with take_time(
            f"Sampling from the target distribution with {SAMPLER} via {SCORING_METRIC}"
        ):
            samples_q = sampler.sample(n_samples, burnin=burnin)
            log_probs = np.array([target_pdf.log_prob(q) for q in samples_q])

        # Save precomputed samples
        os.makedirs("results/reg_3d2d_synthetic", exist_ok=True)
        np.savez(filepath, samples=samples_q, log_probs=log_probs)

    # find the best quaternion
    q_best = samples_q[np.argmax(log_probs)]
    print(f"Best quaternion: {q_best}")

    # transform the source (rotate and project) based on the based quaternion
    transformed_source = source.transform_positions(q_best)

    # fig = plot_registration_results(
    #     target, source, transformed_source, title_prefix="Registration Results"
    # )
    # fig.savefig(
    #     f"results/reg_3d2d_synthetic/{SCORING_METRIC}_{SAMPLER}_seed_{TARGET_SEED}.png",
    #     dpi=150,
    # )

    # Enhanced plot with metrics
    _, reg_error, rot_error = plot_registration_with_metrics(
        target,
        source,
        transformed_source,
        q_best,
        true_rotation,
        scoring_metric=SCORING_METRIC,
        sampler=SAMPLER,
    )

    print(f"Registration Error: {reg_error:.4f}")
    print(f"Rotation Error: {rot_error:.2f}°")


# Generate and visualize the point cloud
if __name__ == "__main__":
    main()
