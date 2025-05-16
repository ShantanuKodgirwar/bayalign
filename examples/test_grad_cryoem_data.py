import os
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import scipy.ndimage as ndi
from jax import random
from scipy.optimize import approx_fprime
from scipy.spatial import KDTree
from scipy.special import logsumexp
from skimage import measure

from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians
from bayalign.sphere_utils import sample_sphere


def load_class_average(index=0, data_path="data/ribosome_80S/pfrib80S_cavgs.mrc"):
    """Loads a projection image (class average) of the 80S ribosome.

    Parameters
    ----------
    index : int, optional
        index for one of the images in the file., by default 0
    data_path : str, optional
        Path to the MRC file.

    Returns
    -------
    np.ndarray
        single image
    """
    assert 0 <= index < 400, f"Index {index} out of bounds (0-399)"

    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    data = mrcfile.open(data_path).data
    return data[index]


def pointcloud_from_class_avg(image, pixelsize=2.68, threshold_method="mean"):
    """Creates a pointcloud from class average projection image

    Parameters
    ----------
    image : np.ndarray
        class average projection image
    pixelsize : float, optional
        Pixel size in Angstrom, by default 2.68
    threshold_method : str, optional
        Method to determine threshold ('mean', 'median', or 'otsu'), by default 'mean'

    Returns
    -------
    target: PointCloud
        Target point cloud
    mask: np.ndarray
        Binary mask of the selected region
    """
    # Determine threshold based on method
    if threshold_method == "mean":
        threshold = np.mean(image)
    elif threshold_method == "median":
        threshold = np.median(image)
    elif threshold_method == "otsu":
        from skimage.filters import threshold_otsu

        threshold = threshold_otsu(image)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")

    mask = image > threshold

    # Find connected regions
    labels = ndi.label(mask)[0]

    # Keep the largest connected region
    props = measure.regionprops(labels, intensity_image=image)
    if not props:
        raise ValueError("No connected regions found above threshold")

    _centers = np.array([prop.weighted_centroid for prop in props])
    index = np.argmax([prop.area for prop in props])
    mask = labels == props[index].label

    # Create a point cloud representing the projected ribosome
    positions = np.transpose(np.nonzero(mask)) * pixelsize
    weights = image[mask] - threshold

    # Create point cloud
    target = PointCloud(positions, weights)

    # Shift point cloud's center such that the center of mass is at (0, 0)
    target.transformed(np.eye(2), -target.center_of_mass)

    return target, mask


def fit_model2d(target, K, n_iter=1000, k=100, verbose=False):
    """
    Fits a 2D model (mixture of gaussians) with fixed number of points (RBF kernels) K to weighted 2D point cloud using
    Expectation-Maximization (EM) algorithm. This returns a fixed bandwidth `sigma`.

    Parameters
    ----------
    target : PointCloud
        Point cloud that will be fitted with a smaller (unweighted) point cloud
        by using Expectation Maximization.
    K : int
        Desired number of points or clusters
    n_iter : int, optional
        Number of iterations in expectation maximization, by default 1000
    k : int, optional
        Number of nearest neighbours for assignment, by default 100
    verbose : bool, optional
        Whether to print progress information, by default False

    Returns
    -------
    x : np.ndarray
        Fitted positions
    sigma : float
        Standard deviation of the associated fit
    """
    # Normalize weights
    w = np.asarray(target.weights / target.weights.sum())
    y = np.asarray(target.positions)

    sigma = None

    # Initialize centers by sampling from the target positions based on weights
    x = y[np.random.choice(np.arange(target.size), K, replace=False, p=w)]

    for i in range(n_iter):
        # E-step: softly assign points to centers using KDTree
        tree = KDTree(x)
        d, indices = tree.query(y, k=min(k, x.shape[0]))

        # Initialize sigma if not already set
        if sigma is None:
            sigma = np.sqrt(np.dot(w, np.square(np.min(d, axis=1))))

        # Compute assignment probabilities
        p = -0.5 * d**2 / sigma**2
        p -= logsumexp(p, axis=1)[:, None]
        np.exp(p, out=p)

        # M-step: update sigma
        sigma = np.sqrt(np.sum(w[:, None] * p * np.square(d)) / 2)

        # Update centers (x) based on assignment probabilities
        n = ndi.sum_labels(w[:, None] * p, indices, index=np.arange(K))
        x = np.array(
            [
                ndi.sum_labels((w * yy)[:, None] * p, indices, index=np.arange(K))
                for yy in y.T
            ]
        )
        x = (x / n).T

        if verbose and (i + 1) % (n_iter // 10) == 0:
            print(f"EM iteration {i + 1}/{n_iter}, sigma={sigma:.4f}")

    return x, sigma


def compare_gradients(scorer, rotation, epsilon=1e-6, title="Gradient Comparison"):
    """
    Compare analytical and numerical gradients for a given scorer and rotation.

    Parameters
    ----------
    scorer : object
        Object with log_prob and gradient methods
    rotation : np.ndarray
        Rotation quaternion
    epsilon : float, optional
        Step size for finite differences, by default 1e-6
    title : str, optional
        Title for the results section, by default "Gradient Comparison"

    Returns
    -------
    dict
        Dictionary with comparison results
    """
    print(f"\n{title}")
    print("-" * len(title))

    t0 = time()
    analytical_grad = scorer.gradient(rotation)
    analytical_time = time() - t0
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Computation time: {analytical_time:.4f} s")

    t0 = time()
    numerical_grad = approx_fprime(rotation, scorer.log_prob, epsilon)
    numerical_time = time() - t0
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Computation time: {numerical_time:.4f} s")

    abs_diff = np.abs(analytical_grad - numerical_grad)
    relative_error = np.linalg.norm(analytical_grad - numerical_grad) / np.linalg.norm(
        numerical_grad
    )
    cosine_similarity = np.dot(analytical_grad, numerical_grad) / (
        np.linalg.norm(analytical_grad) * np.linalg.norm(numerical_grad)
    )

    print(f"Absolute differences: {abs_diff}")
    print(f"Max abs difference: {np.max(abs_diff):.6e}")
    print(f"Relative error: {relative_error:.6e}")
    print(f"Cosine similarity: {cosine_similarity:.6f}")
    print(f"Speed-up: {numerical_time / analytical_time:.1f}x")

    # Criteria for a successful test
    is_accurate = relative_error < 0.1  # 10% tolerance
    has_same_direction = cosine_similarity > 0.9  # cosine similarity > 0.9
    is_faster = analytical_time < numerical_time  # should be faster

    print(f"Test {'PASSED' if is_accurate and has_same_direction else 'FAILED'}")

    results = {
        "analytical_grad": analytical_grad,
        "numerical_grad": numerical_grad,
        "abs_diff": abs_diff,
        "relative_error": relative_error,
        "cosine_similarity": cosine_similarity,
        "analytical_time": analytical_time,
        "numerical_time": numerical_time,
        "is_accurate": is_accurate,
        "has_same_direction": has_same_direction,
        "is_faster": is_faster,
    }

    return results


def test_multiple_rotations(scorer, n_rotations=5, seed=1234):
    """
    Test gradient computation for multiple random rotations.

    Parameters
    ----------
    scorer : object
        Object with log_prob and gradient methods
    n_rotations : int, optional
        Number of rotations to test, by default 5
    seed : int, optional
        Random seed, by default 1234

    Returns
    -------
    list
        List of comparison results dictionaries
    """
    all_results = []

    # Initialize JAX PRNG key
    key = jax.random.PRNGKey(seed)

    print(f"\nTesting {n_rotations} random rotations")
    print("=" * 50)

    for i in range(n_rotations):
        # Generate a new key for each rotation
        key, subkey = jax.random.split(key)

        # Generate random quaternion using JAX
        quat = jax.random.normal(subkey, shape=(4,))
        rotation = quat / jnp.linalg.norm(quat)

        print(f"\nRotation {i + 1}: {rotation}")

        result = compare_gradients(
            scorer, rotation, title=f"Rotation {i + 1}/{n_rotations}"
        )
        all_results.append(result)

    # Summarize results
    print("\nSummary of results:")
    print("=" * 50)

    passed = sum(r["is_accurate"] and r["has_same_direction"] for r in all_results)
    print(f"Passed: {passed}/{n_rotations} tests")

    avg_rel_error = np.mean([r["relative_error"] for r in all_results])
    print(f"Average relative error: {avg_rel_error:.6e}")

    avg_cosine = np.mean([r["cosine_similarity"] for r in all_results])
    print(f"Average cosine similarity: {avg_cosine:.6f}")

    avg_speedup = np.mean(
        [r["numerical_time"] / r["analytical_time"] for r in all_results]
    )
    print(f"Average speed-up: {avg_speedup:.1f}x")

    return all_results


def visualize_pointclouds(
    target, source, rotation=None, sigma=None, title="Point Clouds Visualization"
):
    """
    Visualize 2D point clouds with optional rotation of the source.

    Parameters
    ----------
    target : PointCloud
        Target point cloud (2D)
    source : PointCloud or RotationProjection
        Source point cloud (3D)
    rotation : np.ndarray, optional
        Rotation quaternion to apply to source, by default None
    sigma : float, optional
        Sigma value for visualization, by default None
    title : str, optional
        Plot title, by default "Point Clouds Visualization"
    """
    plt.figure(figsize=(10, 8))

    # Plot target points
    plt.scatter(
        target.positions[:, 0],
        target.positions[:, 1],
        s=target.weights * 20,
        c="blue",
        alpha=0.5,
        label="Target",
    )

    # Transform and plot source points if rotation is provided
    if rotation is not None:
        transformed_positions = source.transform_positions(rotation)
        plt.scatter(
            transformed_positions[:, 0],
            transformed_positions[:, 1],
            s=source.weights * 20,
            c="red",
            alpha=0.5,
            label="Source (rotated)",
        )

    # Draw a circle with radius sigma if provided
    if sigma is not None:
        circle = plt.Circle(
            (0, 0),
            sigma,
            fill=False,
            linestyle="--",
            color="gray",
            label=f"σ = {sigma:.2f}",
        )
        plt.gca().add_patch(circle)

    plt.xlabel("X (Å)")
    plt.ylabel("Y (Å)")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()


def main(class_avg_idx=10, n_particles=2000, test_rotations=3, use_kdtree=False):
    """
    Main function to run the full test pipeline.

    Parameters
    ----------
    class_avg_idx : int, optional
        Index of class average to use, by default 10
    n_particles : int, optional
        Number of particles in 3D model, by default 2000
    test_rotations : int, optional
        Number of rotations to test, by default 3
    use_kdtree : bool, optional
        Whether to use KDTree orbrute force method (for JAX compatibility),
        by default False
    """
    print("=" * 80)
    print("Testing gradients for registration scores with real data")
    print(f"Class average index: {class_avg_idx}, Model particles: {n_particles}")
    print("=" * 80)

    # Load class average and create target point cloud
    try:
        image = load_class_average(class_avg_idx)
        target_cloud, mask = pointcloud_from_class_avg(image)
        print(f"Created target point cloud with {target_cloud.size} points")
    except Exception as e:
        print(f"Error loading class average: {e}")
        raise

    # Load 3D model
    try:
        model_path = f"data/ribosome_80S/model3d_{n_particles}.npz"
        model_3d = np.load(model_path)
        source_cloud = RotationProjection(model_3d["positions"], model_3d["weights"])
        source_cloud.positions -= source_cloud.center_of_mass
        print(f"Loaded 3D model with {source_cloud.size} points")
    except Exception as e:
        print(f"Error loading 3D model: {e}")
        raise

    # Fit 2D model to estimate sigma
    print("\nFitting 2D model to estimate sigma...")
    target_fit2d, sigma = fit_model2d(
        target_cloud, n_particles, n_iter=100, k=50, verbose=True
    )
    print(f"Estimated sigma: {sigma:.4f}")

    # Create scoring functions
    print("\nInitializing scoring functions...")

    # Define parameters for scorers
    k = 20
    beta = 20.0

    log_density_kc = KernelCorrelation(
        target_cloud, source_cloud, sigma, k=k, beta=beta, use_kdtree=use_kdtree
    )
    print(
        f"Created KernelCorrelation scorer with sigma={sigma:.4f}, k={k}, beta={beta}"
    )

    log_density_msg = MixtureSphericalGaussians(
        target_cloud, source_cloud, sigma, k=k, beta=beta, use_kdtree=use_kdtree
    )
    print(
        f"Created MixtureSphericalGaussians scorer with sigma={sigma:.4f}, k={k}, beta={beta}"
    )

    # Test KernelCorrelation gradient
    print("\nTesting KernelCorrelation gradients...")
    kc_results = test_multiple_rotations(log_density_kc, n_rotations=test_rotations)

    # Test MixtureSphericalGaussians gradient
    print("\nTesting MixtureSphericalGaussians gradients...")
    msg_results = test_multiple_rotations(log_density_msg, n_rotations=test_rotations)

    # Visualize one example
    best_rotation_idx = np.argmax([r["cosine_similarity"] for r in kc_results])
    key = random.PRNGKey(1234 + best_rotation_idx)
    best_rotation = sample_sphere(key, d=3)

    print(f"\nVisualizing point clouds with rotation {best_rotation_idx + 1}...")
    _fig = visualize_pointclouds(
        target_cloud,
        source_cloud,
        rotation=best_rotation,
        sigma=sigma,
        title=f"Point Cloud Registration (Sigma={sigma:.2f})",
    )
    plt.show()

    # Print final summary
    kc_passed = sum(r["is_accurate"] and r["has_same_direction"] for r in kc_results)
    msg_passed = sum(r["is_accurate"] and r["has_same_direction"] for r in msg_results)

    print("\nFinal Results Summary:")
    print("=" * 80)
    print(f"KernelCorrelation: {kc_passed}/{test_rotations} tests passed")
    print(f"MixtureSphericalGaussians: {msg_passed}/{test_rotations} tests passed")

    # Calculate average metrics
    kc_avg_rel_error = np.mean([r["relative_error"] for r in kc_results])
    msg_avg_rel_error = np.mean([r["relative_error"] for r in msg_results])

    kc_avg_cosine = np.mean([r["cosine_similarity"] for r in kc_results])
    msg_avg_cosine = np.mean([r["cosine_similarity"] for r in msg_results])

    kc_avg_speedup = np.mean(
        [r["numerical_time"] / r["analytical_time"] for r in kc_results]
    )
    msg_avg_speedup = np.mean(
        [r["numerical_time"] / r["analytical_time"] for r in msg_results]
    )

    print(
        f"KernelCorrelation avg error: {kc_avg_rel_error:.6e}, cosine: {kc_avg_cosine:.4f}, speedup: {kc_avg_speedup:.1f}x"
    )
    print(
        f"MixtureSphericalGaussians avg error: {msg_avg_rel_error:.6e}, cosine: {msg_avg_cosine:.4f}, speedup: {msg_avg_speedup:.1f}x"
    )

    if kc_passed == test_rotations and msg_passed == test_rotations:
        print("\nALL TESTS PASSED! Gradients are working correctly.")
    else:
        print("\nSome tests FAILED. Check individual results for details.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test registration score gradients with real data"
    )
    parser.add_argument("--class_idx", type=int, default=10, help="Class average index")
    parser.add_argument(
        "--particles", type=int, default=2000, help="Number of particles in 3D model"
    )
    parser.add_argument(
        "--rotations", type=int, default=3, help="Number of rotations to test"
    )
    parser.add_argument(
        "--use_kdtree",
        type=bool,
        default=False,
        help="Use KD Tree method",
    )

    args = parser.parse_args()

    main(
        class_avg_idx=args.class_idx,
        n_particles=args.particles,
        test_rotations=args.rotations,
        use_kdtree=args.use_kdtree,
    )
