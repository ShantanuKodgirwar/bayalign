"""
Minimal test script to isolate JAX autodiff gradient issues with CryoEM data.
"""

# TODO: With real data, there are some underflow, overflow issues especially with MixtureSphericalGaussians, need to fix it first.

import os
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, random
from scipy.optimize import approx_fprime

from bayalign.cryo_utils import (
    fit_model2d,
    load_class_average,
    pointcloud_from_class_avg,
)
from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians

# Enable double precision for better numerical stability
jax.config.update("jax_enable_x64", True)


def load_target_cloud(class_idx=10, n_particles=2000, return_fitted_target=False):
    # convert a class average to a point cloud
    image = load_class_average(class_idx)
    target_cloud, _ = pointcloud_from_class_avg(image)

    # find a 2D fit of this target and estimate an optimal sigma
    target_fit2d, sigma = fit_model2d(
        target_cloud,
        n_particles,
        n_iter=100,
        k=50,
        verbose=True,
    )

    if return_fitted_target:
        # converts to a PointCloud class with weights=1.0
        return PointCloud(target_fit2d, weights=None), sigma
    else:
        return target_cloud, sigma


def load_synthetic_target_from_source(source_cloud):
    # Project source with a random rotation to create target
    key = random.PRNGKey(42)
    quat = random.normal(key, shape=(4,))
    rotation = quat / jnp.linalg.norm(quat)

    # Add some noise to make it realistic
    transformed_pos = source_cloud.transform_positions(rotation)

    key, subkey = random.split(key)
    noise = 0.1 * random.normal(subkey, shape=transformed_pos.shape)
    noisy_pos = transformed_pos + noise

    # Create target point cloud
    target_cloud = PointCloud(noisy_pos, source_cloud.weights)

    # Use a reasonable sigma value based on the data
    sigma = 2.0

    return target_cloud, sigma


def load_cryo3D2D(class_idx=10, n_particles=2000, is_synthetic_target=False):
    """
    Load CryoEM data and prepare point clouds.

    Parameters
    ----------
    class_idx : int
        Index of class average to use
    n_particles : int
        Number of particles in the model

    Returns
    -------
    target_cloud : PointCloud
        Target 2D point cloud
    source_cloud : RotationProjection
        Source 3D point cloud for projection
    sigma : float
        Estimated sigma value
    """
    print(f"Loading CryoEM data: class_idx={class_idx}, particles={n_particles}")

    # Load model data
    model_path = f"data/ribosome_80S/model3d_{n_particles}.npz"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_3d = np.load(model_path)
    source_cloud = RotationProjection(model_3d["positions"], model_3d["weights"])
    source_cloud.positions -= source_cloud.center_of_mass

    # For this minimal test, let's use a simple target cloud derived from the source
    # to avoid loading the full CryoEM dataset

    # Transform and project to 2D
    if is_synthetic_target:
        target_cloud, sigma = load_synthetic_target_from_source(source_cloud)
    else:
        target_cloud, sigma = load_target_cloud(
            class_idx, n_particles, return_fitted_target=True
        )

    print(f"Created source cloud with {source_cloud.size} points")
    print(f"Created target cloud with {target_cloud.size} points")
    print(f"Using sigma = {sigma}")

    return target_cloud, source_cloud, sigma


def load_cryo3D3D(n_particles=2000):
    """
    Load CryoEM 3D source and access a 3D target that is a rotated source.

    Parameters
    ----------
    class_idx : int
        Index of class average to use
    n_particles : int
        Number of particles in the model

    Returns
    -------
    target_cloud : PointCloud
        Target 3D point cloud
    source_cloud : PointCloud
        Source 3D point cloud
    sigma : float
        Estimated sigma value
    """

    # Load model data
    model_path = f"data/ribosome_80S/model3d_{n_particles}.npz"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_3d = np.load(model_path)
    source_cloud = PointCloud(model_3d["positions"], model_3d["weights"])
    source_cloud.positions -= source_cloud.center_of_mass

    # For this minimal test, let's use a simple target cloud derived from the source
    # to avoid loading the full CryoEM dataset

    # create a target cloud
    target_cloud, sigma = load_synthetic_target_from_source(source_cloud)

    print(f"Created source cloud with {source_cloud.size} points")
    print(f"Created target cloud with {target_cloud.size} points")
    print(f"Using sigma = {sigma}")

    return target_cloud, source_cloud, sigma


def compare_gradients(scorer, rotation, epsilon=1e-6):
    """
    Compare automatic (AD), analytical, and numerical gradients.

    Parameters
    ----------
    scorer : KernelCorrelation or MixtureSphericalGaussians
        Scoring function
    rotation : jnp.ndarray
        Rotation quaternion
    epsilon : float
        Step size for numerical gradient

    Returns
    -------
    dict
        Comparison results
    """
    results = {}

    # Ensure rotation is a JAX array
    rotation = jnp.array(rotation)

    # Test log_prob first
    try:
        log_prob = scorer.log_prob(rotation)
        print(f"Log probability: {log_prob}")
        if jnp.isnan(log_prob):
            print("WARNING: Log probability is NaN")
        results["log_prob"] = log_prob
    except Exception as e:
        print(f"Error computing log probability: {e}")
        results["log_prob_error"] = str(e)

    # 1. Test automatic differentiation (JAX autodiff)
    print("\n1. Testing automatic differentiation (JAX)...")
    try:
        t0 = time()
        auto_grad = scorer.gradient(rotation)
        auto_time = time() - t0

        print(f"Auto gradient: {auto_grad}")
        print(f"Computation time: {auto_time:.4f}s")

        if jnp.any(jnp.isnan(auto_grad)):
            print("WARNING: Automatic gradient contains NaN values")

        results["auto_grad"] = auto_grad
        results["auto_time"] = auto_time
    except Exception as e:
        print(f"Error computing automatic gradient: {e}")
        results["auto_grad_error"] = str(e)

    # 2. Test numerical gradient (scipy.optimize.approx_fprime)
    print("\n2. Testing numerical gradient (scipy)...")
    try:
        # Define log_prob wrapper for approx_fprime
        def log_prob_wrapper(x):
            return float(scorer.log_prob(jnp.array(x)))

        t0 = time()
        numerical_grad = approx_fprime(np.array(rotation), log_prob_wrapper, epsilon)
        numerical_time = time() - t0

        print(f"Numerical gradient: {numerical_grad}")
        print(f"Computation time: {numerical_time:.4f}s")

        if np.any(np.isnan(numerical_grad)):
            print("WARNING: Numerical gradient contains NaN values")

        results["numerical_grad"] = numerical_grad
        results["numerical_time"] = numerical_time
    except Exception as e:
        print(f"Error computing numerical gradient: {e}")
        results["numerical_grad_error"] = str(e)

    # Compare gradients if both are available
    if "auto_grad" in results and "numerical_grad" in results:
        auto_grad = results["auto_grad"]
        numerical_grad = results["numerical_grad"]

        if not jnp.any(jnp.isnan(auto_grad)) and not np.any(np.isnan(numerical_grad)):
            # Compute difference metrics
            abs_diff = jnp.abs(auto_grad - numerical_grad)
            rel_error = float(
                jnp.linalg.norm(auto_grad - numerical_grad)
                / (jnp.linalg.norm(numerical_grad) + 1e-10)
            )

            # cosine between the two gradients. 1 = same direction,
            # 0 = perpendicular and -1 = opposite
            cos_sim = float(
                jnp.dot(auto_grad, numerical_grad)
                / (jnp.linalg.norm(auto_grad) * jnp.linalg.norm(numerical_grad) + 1e-10)
            )

            print("\nGradient comparison:")
            print(f"Max absolute difference: {float(jnp.max(abs_diff)):.6e}")
            print(f"Relative error: {rel_error:.6e}")
            print(f"Cosine similarity: {cos_sim:.6f}")

            # Store comparison metrics
            results["abs_diff"] = abs_diff
            results["rel_error"] = rel_error
            results["cos_sim"] = cos_sim

            # Check if gradients are close
            is_close = rel_error < 0.1 and cos_sim > 0.9
            print(f"Gradients are {'close' if is_close else 'not close'}")
            results["is_close"] = is_close

    return results


def test_gradient_computation(target, source, sigma):
    """
    Test gradient computation with CryoEM data.
    """
    print("=" * 70)
    print("Testing JAX autodiff gradient computation with CryoEM data")
    print("=" * 70)

    # Load CryoEM data

    # Create scoring functions
    k = 20
    beta = 1.0  # Use a smaller beta for more stability

    # Create KernelCorrelation with brute force
    print("\nCreating KernelCorrelation scorer...")
    kc = KernelCorrelation(target, source, sigma, k=k, beta=beta, use_kdtree=False)

    # Create MixtureSphericalGaussians with brute force
    print("\nCreating MixtureSphericalGaussians scorer...")
    msg = MixtureSphericalGaussians(
        target, source, sigma, k=k, beta=beta, use_kdtree=False
    )

    # Generate random test rotations
    print("\nGenerating test rotations...")
    key = random.PRNGKey(1)
    n_rotations = 3

    rotation_results = []

    for i in range(n_rotations):
        print(f"\n{'-' * 50}")
        print(f"Test rotation {i + 1}/{n_rotations}")
        print(f"{'-' * 50}")

        # Generate random quaternion
        key, subkey = random.split(key)
        quat = random.normal(subkey, shape=(4,))
        rotation = quat / jnp.linalg.norm(quat)

        print(f"Rotation: {rotation}")

        # Test KernelCorrelation
        print("\nTesting KernelCorrelation:")
        kc_results = compare_gradients(kc, rotation)

        # Test MixtureSphericalGaussians
        print("\nTesting MixtureSphericalGaussians:")
        msg_results = compare_gradients(msg, rotation)

        rotation_results.append(
            {"rotation": rotation, "kc_results": kc_results, "msg_results": msg_results}
        )

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    kc_auto_nans = 0
    kc_num_nans = 0
    msg_auto_nans = 0
    msg_num_nans = 0

    for i, res in enumerate(rotation_results):
        kc_res = res["kc_results"]
        msg_res = res["msg_results"]

        print(f"\nRotation {i + 1}:")

        # Check KernelCorrelation
        if "auto_grad" in kc_res:
            auto_nan = jnp.any(jnp.isnan(kc_res["auto_grad"]))
            if auto_nan:
                kc_auto_nans += 1
                print("  KC auto grad: \u274c")
            else:
                print("  KC auto grad: \u2705")
        else:
            kc_auto_nans += 1
            print("  KC auto grad: Error")

        if "numerical_grad" in kc_res:
            num_nan = np.any(np.isnan(kc_res["numerical_grad"]))
            if num_nan:
                kc_num_nans += 1
                print("  KC num grad: \u274c")
            else:
                print("  KC num grad: \u2705")
        else:
            kc_num_nans += 1
            print("  KC num grad: Error")

        # Check MixtureSphericalGaussians
        if "auto_grad" in msg_res:
            auto_nan = jnp.any(jnp.isnan(msg_res["auto_grad"]))
            if auto_nan:
                msg_auto_nans += 1
                print("  MSG auto grad: \u274c")
            else:
                print("  MSG auto grad: \u2705")
        else:
            msg_auto_nans += 1
            print("  MSG auto grad: Error")

        if "numerical_grad" in msg_res:
            num_nan = np.any(np.isnan(msg_res["numerical_grad"]))
            if num_nan:
                msg_num_nans += 1
                print("  MSG num grad: \u274c")
            else:
                print("  MSG num grad: \u2705")
        else:
            msg_num_nans += 1
            print("  MSG num grad: Error")

    print("\nOverall results:")
    print(f"KernelCorrelation auto gradients with NaNs: {kc_auto_nans}/{n_rotations}")
    print(
        f"KernelCorrelation numerical gradients with NaNs: {kc_num_nans}/{n_rotations}"
    )
    print(
        f"MixtureSphericalGaussians auto gradients with NaNs: {msg_auto_nans}/{n_rotations}"
    )
    print(
        f"MixtureSphericalGaussians numerical gradients with NaNs: {msg_num_nans}/{n_rotations}"
    )

    success = (
        kc_auto_nans == 0
        and kc_num_nans == 0
        and msg_auto_nans == 0
        and msg_num_nans == 0
    )

    if success:
        print("\nALL TESTS PASSED: Gradients work correctly with CryoEM data")
    else:
        print("\nSOME TESTS FAILED: Check detailed results")


def debug_jax_grad(target, source, sigma):
    """
    Debug JAX gradient computation with a simplified function.
    """
    print("\n" + "=" * 50)
    print("DEBUGGING JAX GRADIENT COMPUTATION")
    print("=" * 50)

    # Create a simplified version of the log_prob function
    def simplified_log_prob(rotation):
        # Convert to rotation matrix
        from bayalign.utils import quat2matrix

        R = quat2matrix(rotation)

        # Transform source points
        transformed = source.transform_positions(R)

        # Compute distances to first 100 target points
        target_sample = jnp.array(target.positions[:100])
        target_weights = jnp.array(target.weights[:100])

        # Compute all pairwise distances (simplified)
        diff = target_sample[:, None, :] - transformed[None, :, :]
        distances_sq = jnp.sum(diff**2, axis=-1)

        # Compute kernel values
        kernel = jnp.exp(-0.5 * distances_sq / sigma**2)

        # Weight the kernel values
        weighted_kernel = kernel * target_weights[:, None] * source.weights[None, :]

        # Compute log sum
        return jnp.log(jnp.sum(weighted_kernel) + 1e-10)

    # Generate a random rotation
    key = random.PRNGKey(42)
    quat = random.normal(key, shape=(4,))
    rotation = quat / jnp.linalg.norm(quat)

    # Test function value
    print("\nTesting simplified log_prob function...")
    try:
        value = simplified_log_prob(rotation)
        print(f"Function value: {value}")

        if jnp.isnan(value):
            print("WARNING: Function value is NaN")
    except Exception as e:
        print(f"Error computing function value: {e}")

    # Test JAX autodiff gradient
    print("\nTesting JAX autodiff gradient of simplified function...")
    try:
        grad_fn = grad(simplified_log_prob)
        auto_grad = grad_fn(rotation)
        print(f"JAX autodiff gradient: {auto_grad}")

        if jnp.any(jnp.isnan(auto_grad)):
            print("WARNING: JAX autodiff gradient contains NaN values")
    except Exception as e:
        print(f"Error computing JAX autodiff gradient: {e}")

    # Test numerical gradient
    print("\nTesting numerical gradient of simplified function...")
    try:
        numerical_grad = approx_fprime(
            np.array(rotation),
            lambda x: float(simplified_log_prob(jnp.array(x))),
            epsilon=1e-6,
        )
        print(f"Numerical gradient: {numerical_grad}")

        if np.any(np.isnan(numerical_grad)):
            print("WARNING: Numerical gradient contains NaN values")
    except Exception as e:
        print(f"Error computing numerical gradient: {e}")


if __name__ == "__main__":
    # load point cloud data
    if False:
        # uses a cryoEM projection image (fitted to a pointcloud) as 2D target
        # and 3D model as source
        target, source, sigma = load_cryo3D2D(
            class_idx=10,
            n_particles=2000,
            is_synthetic_target=False,
        )
    else:
        # loads a synthetic 3D target made from 3D source
        target, source, sigma = load_cryo3D3D(
            n_particles=2000,
        )

    # Test gradient computation
    test_gradient_computation(target, source, sigma)

    # Debug JAX gradient computation
    debug_jax_grad(target, source, sigma)

    print("\n" + "=" * 50)
    print("END OF TESTS")
    print("=" * 50)
