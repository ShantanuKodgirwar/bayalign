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
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.tree_util import Partial
from jaxtyping import Array, Float

from .kdtree import build_tree, query_neighbors

# Import from your pointcloud module
from .pointcloud import PointCloud
from .types import Quaternion
from .utils import matrix2quat, quat2matrix

# Type aliases for clarity
Points2D = Float[Array, "n 2"]
Points3D = Float[Array, "n 3"]
Weights = Float[Array, "n"]
Distances = Float[Array, "n k"]
Indices = Float[Array, "n k"]


@Partial(jax.jit, static_argnums=(2,))
def query_neighbors_pairwise(points, query, k):
    """
    Find the k-nearest neighbors by forming a pairwise distance matrix.
    A brute-force O(N^2) implementation.

    Parameters
    ----------
    points: jnp.ndarray
        (N, d) Points to search.
    query: jnp.ndarray
        (d,) or (Q, d) Query point(s).
    k : int
        Number of neighbors to return.

    Returns
    -------
    jnp.ndarray
        (k,) or (Q, k) Indices of nearest neighbors.
    jnp.ndarray
        (k,) or (Q, k) Distances to nearest neighbors.
    """
    query_shaped = jnp.atleast_2d(query)
    pairwise_distances = jnp.linalg.norm(points - query_shaped[:, None], axis=-1)
    distances, indices = jax.lax.top_k(-1 * pairwise_distances, k)
    if query.ndim == 1:
        return indices.squeeze(0), -1 * distances.squeeze(0)
    return indices, -1 * distances


# ---------------------------------------------------------------------- #
#  Kernel correlation                                                    #
# ---------------------------------------------------------------------- #
class Registration:
    """
    Rigid registration scoring assigning a log-probability to a rigid body pose.
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

    # public API
    @partial(jax.jit, static_argnums=(0,))
    def log_prob(self, rotation, translation=None):
        """Compute log probability for a given rotation."""
        q = self._ensure_quaternion(rotation)
        return self.beta * self._log_prob_impl(q, translation)

    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, rotation, translation=None):
        """Compute gradient with respect to quaternion parameters."""
        q = self._ensure_quaternion(rotation)
        grad_fn = jax.grad(lambda q: self._log_prob_impl(q, translation))
        return self.beta * grad_fn(q)

    def _log_prob_impl(self, q, translation=None):
        """Implementation of log probability calculation."""
        raise NotImplementedError("Subclasses must implement _log_prob_impl")


@dataclass(frozen=True)
class KernelCorrelation(Registration):
    """
    Kernel Correlation scoring method for rigid registration between point clouds.

    This class implements the Kernel Correlation (KC) scoring method, which assigns a
    log-probability score to a rigid body transformation between two point clouds.
    The score is based on the sum of Gaussian kernels between the target points
    and their k-nearest neighbors in the transformed source point cloud.

    The KC score is defined as:
    log p(R) = log sum_{i,j} w_i v_j exp(-||y_i - PRx_j||^2 / (2σ^2))

    where:
    - y_i are the target points with weights w_i
    - x_j are the source points with weights v_j
    - R is the rotation matrix
    - P is the optional projection for 3D->2D
    - σ is the kernel bandwidth parameter

    Parameters
    ----------
    target : PointCloud
        Target point cloud
    source : PointCloud
        Source point cloud to be transformed
    sigma : float, default=1.0
        Bandwidth parameter controlling the width of Gaussian kernels
    beta : float, default=1.0
        Inverse temperature parameter that scales the log probability
    k : int, default=20
        Number of nearest neighbors to consider for each target point
    use_kdtree : bool, default=False
        Whether to use a KDTree for efficient nearest neighbor search

    Notes
    -----
    The implementation is JAX-compatible and supports automatic differentiation
    for gradient computation, which is essential for optimization and sampling.
    """

    target: PointCloud
    source: PointCloud
    sigma: float = 1.0
    beta: float = 1.0
    k: int = 20
    use_kdtree: bool = False

    def __post_init__(self):
        """Precompute constants for efficiency."""
        # Validate inputs
        if self.sigma < 1e-6:
            warnings.warn(
                f"Small sigma value ({self.sigma}) may cause numerical instability"
            )

        # Store frequently accessed values
        object.__setattr__(self, "_sigma", float(self.sigma))
        object.__setattr__(self, "_k", min(int(self.k), self.source.size))
        object.__setattr__(self, "_inv_2sigma_sq", 1.0 / (2.0 * self.sigma**2))

        # Pre-extract arrays to avoid repeated attribute access
        object.__setattr__(self, "_target_pos", self.target.positions)
        object.__setattr__(self, "_target_weights", self.target.weights)
        object.__setattr__(self, "_source_pos", self.source.positions)
        object.__setattr__(self, "_source_weights", self.source.weights)

        # Precompute log weights for efficiency
        object.__setattr__(self, "_log_target_weights", jnp.log(self._target_weights))
        object.__setattr__(self, "_log_source_weights", jnp.log(self._source_weights))

    @partial(jax.jit, static_argnums=(0,))
    def _log_prob_impl(
        self, q: Quaternion, translation: Optional[Array] = None
    ) -> float:
        """
        Implementation of log probability calculation for Kernel Correlation.

        Computes the log probability of a given rotation (represented as a quaternion)
        and optional translation by:
        1. Transforming the source points
        2. Finding k-nearest neighbors for each target point
        3. Computing Gaussian kernel values
        4. Summing the weighted kernel values in log-space

        Parameters
        ----------
        q : jnp.ndarray
            Quaternion representing rotation with shape (4,)
        translation : jnp.ndarray, optional
            Translation vector, by default None

        Returns
        -------
        float
            Log probability score for the given transformation

        Notes
        -----
        This implementation is fully JAX-compatible to support auto-differentiation
        and uses optimized KNN search for efficiency.
        """

        # Transform source points
        R = quat2matrix(q)
        src_pos_transformed = self.source.transform_positions(R, translation)

        # Efficient nearest neighbors search
        if self.use_kdtree:
            tree = build_tree(src_pos_transformed)
            neighbors_idx, dists = query_neighbors(
                tree,
                self._target_pos,
                self._k,
            )
        else:
            neighbors_idx, dists = query_neighbors_pairwise(
                self._target_pos,
                src_pos_transformed,
                self._k,
            )

        # Explicitly gather the weights - more clear and sometimes faster
        # indices has shape (n_targets, k)
        selected_log_weights = jnp.take(self._log_source_weights, neighbors_idx, axis=0)

        # Compute the Gaussian kernel
        log_kernel_values = (
            -dists * dists * self._inv_2sigma_sq
            + +selected_log_weights  # shape (n_targets, k)
            + self._log_target_weights[:, None]  # Broadcasting to (n_targets, 1)
        )

        # Return log probability (logsumexp for numerical stability)
        return logsumexp(log_kernel_values)


@dataclass(frozen=True)
class GaussianMixtureModel(Registration):
    """
    Mixture of Spherical Gaussians (MSG) scoring method for rigid registration.

    This class implements the Mixture of Spherical Gaussians scoring method, which
    assigns a log-probability score to a rigid body transformation between point clouds.
    It models each transformed source point as the center of a Gaussian distribution
    and computes the likelihood of target points under this mixture model.

    The MSG score is defined as:
    log p(R) = sum_i w_i log sum_j v_j N(y_i | PRx_j, σ^2I)

    where:
    - y_i are the target points with weights w_i
    - x_j are the source points with weights v_j
    - R is the rotation matrix
    - P is the optional projection for 3D->2D
    - N(y | μ, σ^2I) is a Gaussian with mean μ and covariance σ^2I

    Parameters
    ----------
    target : PointCloud
        Target point cloud
    source : PointCloud
        Source point cloud to be transformed
    sigma : float, default=1.0
        Standard deviation of the spherical Gaussians
    beta : float, default=1.0
        Inverse temperature parameter that scales the log probability
    k : int, default=20
        Number of nearest neighbors to use for efficient computation
    use_kdtree : bool, default=False
        Whether to use a KDTree for efficient nearest neighbors search

    Notes
    -----
    This implementation differs from Kernel Correlation in how the scores are computed.
    MSG computes a weighted sum of log probabilities, while KC computes a log of weighted
    sum of kernel values. MSG tends to be more robust to differences in point densities.
    """

    target: PointCloud
    source: PointCloud
    sigma: float = 1.0
    beta: float = 1.0
    k: int = 20
    use_kdtree: bool = False

    def __post_init__(self):
        """Precompute constants for efficiency."""
        # Validate
        if self.sigma < 1e-6:
            warnings.warn(
                f"Small sigma value ({self.sigma}) may cause numerical instability"
            )

        # Store constants
        object.__setattr__(self, "_sigma", float(self.sigma))
        object.__setattr__(self, "_k", min(int(self.k), self.source.size))
        object.__setattr__(self, "_inv_sigma_sq", 1.0 / (self.sigma**2))

        # Precompute normalization constant
        d = self.target.dim
        object.__setattr__(
            self, "_log_norm", -0.5 * d * jnp.log(2 * jnp.pi * self.sigma**2)
        )

        # Pre-extract arrays
        object.__setattr__(self, "_target_pos", self.target.positions)
        object.__setattr__(self, "_target_weights", self.target.weights)
        object.__setattr__(self, "_source_pos", self.source.positions)
        object.__setattr__(self, "_source_weights", self.source.weights)
        object.__setattr__(self, "_log_source_weights", jnp.log(self._source_weights))

    @partial(jax.jit, static_argnums=(0,))
    def _log_prob_impl(
        self, q: Quaternion, translation: Optional[Array] = None
    ) -> float:
        """
        Implementation of log probability calculation for Mixture of Spherical Gaussians.

        Computes the log probability of a given rotation (represented as a quaternion)
        and optional translation by:
        1. Transforming the source points
        2. Finding k-nearest neighbors for each target point
        3. Computing Gaussian likelihoods
        4. Computing weighted log probabilities for each target point

        Parameters
        ----------
        q : jnp.ndarray
            Quaternion representing rotation with shape (4,)
        translation : jnp.ndarray, optional
            Translation vector, by default None

        Returns
        -------
        float
            Log probability score for the given transformation

        Notes
        -----
        The nearest neighbors approach is used for efficiency, as computing the
        full mixture model would require evaluating all pairs of points, which
        would be computationally prohibitive for large point clouds.
        """

        # Transform source points
        R = quat2matrix(q)
        src_pos_transformed = self.source.transform_positions(R, translation)

        # Efficient nearest neighbors search
        if self.use_kdtree:
            tree = build_tree(src_pos_transformed)
            neighbors_idx, dists = query_neighbors(
                tree,
                self._target_pos,
                self._k,
            )
        else:
            neighbors_idx, dists = query_neighbors_pairwise(
                self._target_pos,
                src_pos_transformed,
                self._k,
            )

        # Compute log probabilities with fused operations
        # log p(y_i | x_j) = log_norm - 0.5 * ||y_i - x_j||^2 / sigma^2 + log(w_j)
        log_probs = (
            self._log_norm
            - 0.5 * dists * dists * self._inv_sigma_sq
            + self._log_source_weights[neighbors_idx]
        )

        # For each target point, compute log sum exp over k nearest sources
        target_log_probs = logsumexp(log_probs, axis=1)

        # Weighted sum over target points
        return jnp.dot(target_log_probs, self._target_weights)
