"""
Test script to validate AD based gradient computation with
finite differences (scipy.optimize.approx_fprime).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.optimize import approx_fprime

# Import your implementation
from bayalign.pointcloud import PointCloud, RotationProjection
from bayalign.score import KernelCorrelation, MixtureSphericalGaussians
from bayalign.utils import quat2matrix

# Set random seed for reproducibility
np.random.seed(42)
key = random.PRNGKey(42)
jax.config.update("jax_enable_x64", True)  # Use double precision for stability


def random_rotation(key):
    """Generate a random rotation as a quaternion (x, y, z, w)."""
    key, subkey = random.split(key)
    quat = random.normal(subkey, shape=(4,))
    quat = quat / jnp.linalg.norm(quat)
    return quat


def create_test_point_clouds(key, n_points=100, dim=3, noise_level=0.1):
    """Create test point clouds for registration testing."""
    key, subkey1, subkey2 = random.split(key, 3)
    positions = random.normal(subkey1, shape=(n_points, dim))
    weights = jnp.ones(n_points) / n_points

    source_pc = PointCloud(positions, weights)

    true_rotation = random_rotation(subkey2)
    rotation_matrix = quat2matrix(true_rotation)

    rotated_positions = positions @ rotation_matrix.T
    key, noise_key = random.split(key)
    noise = noise_level * random.normal(noise_key, shape=(n_points, dim))
    noisy_positions = rotated_positions + noise

    target_pc = PointCloud(noisy_positions, weights)

    return source_pc, target_pc, true_rotation


def compute_numerical_gradient(scorer, rotation, epsilon=1e-6):
    """
    Compute numerical gradient of log_prob with respect to rotation quaternion
    using finite differences.

    Parameters:
        scorer: Object with log_prob method
        rotation: Quaternion array of shape (4,)
        epsilon: Step size for finite differences

    Returns:
        Numerical gradient array of shape (4,)
    """
    # Convert to numpy for scipy's approx_fprime
    rotation_np = np.array(rotation)

    # Define function that takes numpy array and returns scalar
    def f(x):
        return float(scorer.log_prob(jnp.array(x)))

    # Compute numerical gradient
    num_grad = approx_fprime(rotation_np, f, epsilon)

    return jnp.array(num_grad)


def test_gradient_flow():
    """Test if gradients flow correctly through the scoring functions."""
    print("Testing gradient flow...")

    global key

    # Create 3D test data
    key, subkey1 = random.split(key)
    source_3d, target_3d, true_rotation = create_test_point_clouds(
        subkey1, n_points=100, dim=3
    )

    # Create 2D test data (for 3D-2D projection test)
    key, subkey2, subkey3 = random.split(key, 3)
    source_3d_for_proj, _, _ = create_test_point_clouds(subkey2, n_points=100, dim=3)
    projection_source = RotationProjection(
        source_3d_for_proj.positions, source_3d_for_proj.weights
    )

    positions_2d = random.normal(subkey3, shape=(80, 2))
    weights_2d = jnp.ones(80) / 80
    target_2d = PointCloud(positions_2d, weights_2d)

    key, subkey4 = random.split(key)
    test_rotation = random_rotation(subkey4)

    # Test KernelCorrelation with brute force in 3D-3D scenario
    print("\n1. Testing KernelCorrelation (3D-3D) with brute force...")
    kc_3d = KernelCorrelation(target_3d, source_3d, sigma=0.5, k=10, use_kdtree=False)

    log_prob_3d = kc_3d.log_prob(test_rotation)
    print(f"  Log probability: {log_prob_3d}")

    try:
        # Automatic gradient
        grad_3d = kc_3d.gradient(test_rotation)
        print(f"  Automatic gradient: {grad_3d}")

        # Numerical gradient
        num_grad_3d = compute_numerical_gradient(kc_3d, test_rotation)
        print(f"  Numerical gradient: {num_grad_3d}")

        # Compute relative difference
        grad_diff = jnp.linalg.norm(grad_3d - num_grad_3d)
        grad_rel_diff = grad_diff / (jnp.linalg.norm(grad_3d) + 1e-10)
        print(f"  Gradient difference (L2 norm): {grad_diff}")
        print(f"  Relative difference: {grad_rel_diff:.6f}")

        # Check if gradients are close enough
        is_close = grad_rel_diff < 0.1  # 10% tolerance
        print(
            f"  {'\u2705' if is_close else '\u274c'} Gradients are {'close' if is_close else 'not close'}"
        )

        print("  \u2705 Gradient computation successful!")
    except Exception as e:
        print(f"  \u274c Gradient computation failed: {e}")

    # Test MixtureSphericalGaussians with brute force in 3D-2D scenario
    print("\n2. Testing MixtureSphericalGaussians (3D-2D) with brute force...")
    msg_3d2d = MixtureSphericalGaussians(
        target_2d, projection_source, sigma=0.5, k=10, use_kdtree=False
    )

    log_prob_3d2d = msg_3d2d.log_prob(test_rotation)
    print(f"  Log probability: {log_prob_3d2d}")

    try:
        # Automatic gradient
        grad_3d2d = msg_3d2d.gradient(test_rotation)
        print(f"  Automatic gradient: {grad_3d2d}")

        # Numerical gradient
        num_grad_3d2d = compute_numerical_gradient(msg_3d2d, test_rotation)
        print(f"  Numerical gradient: {num_grad_3d2d}")

        # Compute relative difference
        grad_diff_3d2d = jnp.linalg.norm(grad_3d2d - num_grad_3d2d)
        grad_rel_diff_3d2d = grad_diff_3d2d / (jnp.linalg.norm(grad_3d2d) + 1e-10)
        print(f"  Gradient difference (L2 norm): {grad_diff_3d2d}")
        print(f"  Relative difference: {grad_rel_diff_3d2d:.6f}")

        # Check if gradients are close enough
        is_close_3d2d = grad_rel_diff_3d2d < 0.1  # 10% tolerance
        print(
            f"  {'\u2705' if is_close_3d2d else '\u274c'} Gradients are {'close' if is_close_3d2d else 'not close'}"
        )

        print("  \u2705 Gradient computation successful!")
    except Exception as e:
        print(f"  \u274c Gradient computation failed: {e}")

    # Check optimization direction - do gradients point toward true rotation?
    q_diff = test_rotation - true_rotation
    dot_product = jnp.dot(grad_3d, -q_diff)  # Negative because we're maximizing
    print("\nOptimization direction check:")
    print(f"  Dot product with direction to true rotation: {dot_product}")
    print(
        f"  {'\u2705' if dot_product > 0 else '\u274c'} Gradient points {'toward' if dot_product > 0 else 'away from'} the true rotation"
    )

    # Summary
    if is_close and is_close_3d2d:
        print(
            "\n\u2705 GRADIENT VALIDATION PASSED: Automatic gradients match numerical approximations!"
        )
    else:
        issues = []
        if not is_close:
            issues.append(
                "KernelCorrelation gradients don't match numerical approximation"
            )
        if not is_close_3d2d:
            issues.append(
                "MixtureSphericalGaussians gradients don't match numerical approximation"
            )
        print(f"\n\u274c GRADIENT VALIDATION FAILED: {', '.join(issues)}")


if __name__ == "__main__":
    test_gradient_flow()
