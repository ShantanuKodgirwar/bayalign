"""
Rotation-dependent scores for 3D-2D rigid registration log densities implemented in JAX.

All classes implement the mini-interface required by the MCMC samplers for inference:

    .log_prob(rotation, translation=None)   > float
    .gradient(rotation, translation=None)   > jnp.ndarray shape (4,)   (∂/∂ quaternion)

`rotation` can be either:
    - a unit quaternion (x, y, z, w) with shape (4,)
    - a rotation matrix with shape (3, 3)

Gradient is always returned with respect to quaternion parameters, computed via JAX autodiff.

NOTE: Currently we are only interested in sampling rotations and ignore translations, so `translation=None`
is set everywhere. If needed, we could add support for translations in the future.
"""

import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.scipy.special import logsumexp
from scipy.spatial import KDTree

# Import from your pointcloud module
from .pointcloud import PointCloud
from .utils import matrix2quat, quat2matrix

# Define the integer type to be used for indices
int_type = np.int32


@partial(jit, static_argnames=("k", "pc_threshold"))
def find_nearest_k_indices(
    y: jnp.ndarray,  # (L, dim)
    x: jnp.ndarray,  # (K, dim)
    k: int,
    pc_threshold: int = 5000,
) -> jnp.ndarray:
    """
    Find the indices of the k nearest neighbors from x for each point in y.
    Only returns indices, not distances.
    """
    # Adjust k to avoid out-of-bounds errors
    k = min(k, x.shape[0])

    # All pair-wise squared distances (L, K)
    d2 = jnp.sum((y[:, None, :] - x[None, :, :]) ** 2, axis=-1)

    # Get indices of k smallest distances (use negative for top_k)
    _, idx = jax.lax.top_k(-d2, k)

    if y.shape[0] > pc_threshold or x.shape[0] > pc_threshold:
        warnings.warn("Large input size may impact performance")

    return idx


def kd_tree_nn(points: jax.Array, test_points: jax.Array, k: int = 1) -> jax.Array:
    """
    Uses a KD-tree to find the k nearest neighbors to a test point. Implementation
    adapted from https://github.com/jax-ml/jax/discussions/9813#discussioncomment-11513589

    Parameters:
        points: [n, d] Array of points.
        test_points: [m, d] points to query
        k: The number of nearest neighbors to find.

    Returns:
        distances: [m, k] Squared distances to the k nearest neighbors.
        indices: [m, k] Indices of the k nearest neighbors.
    """
    m, d = np.shape(test_points)
    k = int(k)
    args = (points, test_points, k)

    index_shape_dtype = jax.ShapeDtypeStruct(shape=(m, k), dtype=int_type)

    return jax.pure_callback(_kd_tree_idx_host, index_shape_dtype, *args)


def _kd_tree_idx_host(points: jax.Array, test_points: jax.Array, k: int) -> np.ndarray:
    """
    Host function that builds and queries the KD-tree.
    """
    points, test_points = jax.tree.map(np.asarray, (points, test_points))
    k = int(k)
    tree = KDTree(points, compact_nodes=False, balanced_tree=False)
    if k == 1:
        _, indices = tree.query(test_points, k=[1])
        indices = indices.reshape(-1, 1)
    else:
        _, indices = tree.query(test_points, k=k)

    # Return squared distances for consistency with the rest of the code
    return indices.astype(int_type)


# ---------------------------------------------------------------------- #
#  Abstract `Registration` class                                         #
# ---------------------------------------------------------------------- #
class Registration:
    """
    Rigid registration scoring assigning a log-probability to a rigid body pose.
    Concrete subclasses implement the *SO(3)-level* score mainly via Kernel Correlation
    and Log Mixture of Gaussians.

    All methods accept either a rotation matrix or a unit quaternion, and gradients
    are always computed with respect to quaternion parameters using JAX autodiff.
    """

    beta: float = 1.0  # per-metric inverse temperature

    def _is_quaternion(self, rotation):
        """Check if the rotation is a quaternion or a matrix."""
        return rotation.shape == (4,)

    def _ensure_quaternion(self, rotation):
        """Convert rotation matrix to quaternion if needed."""
        return rotation if self._is_quaternion(rotation) else matrix2quat(rotation)

    def _ensure_matrix(self, rotation):
        """Convert quaternion to rotation matrix if needed."""
        return quat2matrix(rotation) if self._is_quaternion(rotation) else rotation

    # public API ----------------------------------------------------------
    def log_prob(self, rotation, translation=None):
        """
        Compute log probability for a given rotation (matrix or quaternion).
        """
        # Internally, use quaternion for consistency with gradient calculation
        q = self._ensure_quaternion(rotation)
        return self.beta * self._log_prob_impl(q, translation)

    def gradient(self, rotation, translation=None):
        """
        Compute gradient of log probability with respect to quaternion parameters
        using JAX autodiff.
        """
        q = self._ensure_quaternion(rotation)
        # Use JAX's autodiff to compute gradient with respect to quaternion
        return self.beta * grad(lambda q: self._log_prob_impl(q, translation))(q)

    def _log_prob_impl(self, q, translation=None):
        """
        Implementation of log probability calculation that subclasses must override.
        This should accept a quaternion (for gradient consistency) but may
        convert to matrix internally if needed.
        """
        raise NotImplementedError("Subclasses must implement _log_prob_impl")


# ---------------------------------------------------------------------- #
#  Kernel correlation                                                    #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class KernelCorrelation(Registration):
    target: PointCloud
    source: PointCloud
    sigma: float = 1.0
    beta: float = 1.0
    k: int = 20
    use_kdtree: bool = True
    pc_threshold: int = 5000

    def __post_init__(self):
        sig = float(self.sigma)
        k = int(self.k)
        use_tree = bool(self.use_kdtree)
        threshold = int(self.pc_threshold)

        # Check for numerical stability with small sigma values
        if sig < 1e-6:
            warnings.warn(f"Small sigma value ({sig}) may cause numerical instability")

        # Store static arrays
        tgt_pos = self.target.positions
        tgt_w = self.target.weights
        src_w = self.source.weights

        @partial(jit, static_argnames=("k", "threshold"))
        def _log_prob_impl(q, translation=None, k=k, threshold=threshold):
            """Log probability implementation with separated discrete and continuous operations."""
            # Convert quaternion to rotation matrix
            R = quat2matrix(q)

            # Transform the source points using the rotation matrix
            src_pos_transformed = self.source.transform_positions(R, translation)

            # Non-differentiable part: Find nearest neighbors for each target point
            if use_tree:
                idx = kd_tree_nn(src_pos_transformed, tgt_pos, k)
            else:
                idx = find_nearest_k_indices(tgt_pos, src_pos_transformed, k, threshold)

            # Stop gradient through the indices
            idx = jax.lax.stop_gradient(idx)

            # Differentiable part: Re-compute distances for gradient flow
            # Get the selected source points
            src_selected = jnp.take(src_pos_transformed, idx, axis=0)

            # Compute squared distances
            d2 = jnp.sum((tgt_pos[:, None, :] - src_selected) ** 2, axis=-1)

            # For kernel correlation, we compute the Gaussian kernel (L, k)
            log_kernel_values = (
                -0.5 * d2 / sig**2
                + jnp.log(jnp.take(src_w, idx))
                + +jnp.log(tgt_w[:, None])
            )

            # return log probability (negative cost) scaled by beta
            return logsumexp(log_kernel_values, axis=None)

        # Store the implementation
        object.__setattr__(self, "_log_prob_impl", _log_prob_impl)


# ---------------------------------------------------------------------- #
#  Mixture of spherical Gaussians (soft-ICP)                             #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class MixtureSphericalGaussians(Registration):
    target: PointCloud
    source: PointCloud  # Can be PointCloud or RotationProjection
    sigma: float = 1.0
    beta: float = 1.0
    k: int = 20
    use_kdtree: bool = True
    pc_threshold: int = 5000  # Threshold for warning about large point clouds
    """
    Log-likelihood of points under a transformed Gaussian mixture.
    Works for 3D-3D, 3D-2D, or 2D-2D transformations.
    """

    def __post_init__(self):
        sig = float(self.sigma)
        k = int(self.k)
        use_tree = bool(self.use_kdtree)
        threshold = int(self.pc_threshold)

        # Check for numerical stability with small sigma values
        if sig < 1e-6:
            warnings.warn(f"Small sigma value ({sig}) may cause numerical instability")

        # Store static arrays
        tgt_pos = self.target.positions
        tgt_w = self.target.weights
        src_w = self.source.weights

        @partial(jit, static_argnames=("k", "threshold"))
        def _log_prob_impl(q, translation=None, k=k, threshold=threshold):
            """Log probability implementation using either KD-tree or brute force."""
            # Convert quaternion to rotation matrix for use with transform_positions
            R = quat2matrix(q)

            # Transform the source points using the rotation matrix
            src_pos_transformed = self.source.transform_positions(R, translation)

            # Non-differentiable part: Find nearest neighbors for each target point
            if use_tree:
                idx = kd_tree_nn(src_pos_transformed, tgt_pos, k)
            else:
                idx = find_nearest_k_indices(tgt_pos, src_pos_transformed, k, threshold)

            # Stop gradient through the indices (ignores this
            # in the computational graph)
            idx = jax.lax.stop_gradient(idx)

            # Differentiable part: Recompute distances for gradient flow
            src_selected = jnp.take(src_pos_transformed, idx, axis=0)
            d2 = jnp.sum((tgt_pos[:, None, :] - src_selected) ** 2, axis=-1)

            # Compute probability
            phi = jnp.exp(-0.5 * d2 / sig**2)
            w_k = src_w[idx]
            ll = logsumexp(
                jnp.log(w_k) + jnp.log(phi) - jnp.log(2 * jnp.pi * sig**2), axis=1
            )

            return jnp.sum(ll * tgt_w)

        # Store the implementation
        object.__setattr__(self, "_log_prob_impl", _log_prob_impl)
