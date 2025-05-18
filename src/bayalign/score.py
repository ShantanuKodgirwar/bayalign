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

# Import from your pointcloud module
from .pointcloud import PointCloud
from .utils import matrix2quat, quat2matrix

# Define the integer type to be used for indices
int_type = np.int32


# TODO: Currently a brute-force KNN implementation (O(N^2))is used as it is pure JAX, however, better methods could be explored later.
@partial(jit, static_argnames=("k",))
def knn_search(target_points, source_points, k):
    """
    Efficient JAX-compatible K-nearest neighbors search.

    Parameters
    ----------
    target_points : jnp.ndarray
        Query points, shape (M, D)
    source_points : jnp.ndarray
        Reference points, shape (N, D)
    k : int
        Number of nearest neighbors to find
        Assumed to be a static value for JIT compilation

    Returns
    -------
    jnp.ndarray
        Indices of nearest neighbors, shape (M, k)
    """
    # Compute squared Euclidean distances
    target_norm2 = jnp.sum(target_points**2, axis=1, keepdims=True)
    source_norm2 = jnp.sum(source_points**2, axis=1)
    dots = jnp.dot(target_points, source_points.T)
    dists2 = target_norm2 - 2 * dots + source_norm2

    # Ensure distances are non-negative
    dists2 = jnp.maximum(dists2, 0.0)

    # Find k smallest distances - k is assumed to be a static argument
    _, indices = jax.lax.top_k(-dists2, k=k)

    # Clip indices to valid range in case k > n_source
    n_source = source_points.shape[0]
    indices = jnp.clip(indices, 0, n_source - 1)

    return indices


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
    def log_prob(self, rotation, translation=None):
        """Compute log probability for a given rotation."""
        q = self._ensure_quaternion(rotation)
        return self.beta * self._log_prob_impl(q, translation)

    def gradient(self, rotation, translation=None):
        """Compute gradient with respect to quaternion parameters."""
        q = self._ensure_quaternion(rotation)
        grad_q = grad(lambda q: self._log_prob_impl(q, translation))(q)
        return self.beta * grad_q

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

    def __post_init__(self):
        """
        Initialize parameters after instance creation.

        Validates inputs, converts parameters to the correct types,
        and issues warnings for potential numerical issues.
        """
        sigma = float(self.sigma)
        k = int(self.k)

        # Check for numerical stability with small sigma values
        if sigma < 1e-6:
            warnings.warn(
                f"Small sigma value ({sigma}) may cause numerical instability"
            )

        # Store parameters
        object.__setattr__(self, "_sigma", sigma)
        object.__setattr__(self, "_k", k)

    def _log_prob_impl(self, q, translation=None):
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
        # Convert quaternion to rotation matrix
        R = quat2matrix(q)

        # Transform source points
        src_pos_transformed = self.source.transform_positions(R, translation)

        # Extract target arrays
        target_pos = self.target.positions
        target_weights = self.target.weights
        source_weights = self.source.weights

        # Efficient nearest neighbors search
        idx = knn_search(target_pos, src_pos_transformed, self._k)

        # Gather selected points and weights
        selected_points = jnp.take(src_pos_transformed, idx, axis=0)
        selected_weights = jnp.take(source_weights, idx, axis=0)

        # Compute squared distances
        d2 = jnp.sum((target_pos[:, None, :] - selected_points) ** 2, axis=-1)

        # Compute the Gaussian kernel
        log_kernel_values = (
            -0.5 * d2 / (self._sigma**2)
            + jnp.log(selected_weights)
            + jnp.log(target_weights[:, None])
        )

        # Return log probability (logsumexp for numerical stability)
        return logsumexp(log_kernel_values, axis=None)


@dataclass(frozen=True)
class MixtureSphericalGaussians(Registration):
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

    def __post_init__(self):
        """
        Initialize parameters after instance creation.

        Validates inputs, converts parameters to the correct types,
        and issues warnings for potential numerical issues.
        """
        sigma = float(self.sigma)
        k = int(self.k)

        # Check for numerical stability with small sigma values
        if sigma < 1e-6:
            warnings.warn(
                f"Small sigma value ({sigma}) may cause numerical instability"
            )

        # Store parameters
        object.__setattr__(self, "_sigma", sigma)
        object.__setattr__(self, "_k", k)

    def _log_prob_impl(self, q, translation=None):
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
        # Convert quaternion to rotation matrix
        R = quat2matrix(q)

        # Transform source points
        src_pos_transformed = self.source.transform_positions(R, translation)

        # Extract target arrays
        target_pos = self.target.positions
        target_weights = self.target.weights
        source_weights = self.source.weights

        # Efficient nearest neighbors search
        idx = knn_search(target_pos, src_pos_transformed, self._k)

        # Gather selected points and weights
        selected_points = jnp.take(src_pos_transformed, idx, axis=0)
        selected_weights = jnp.take(source_weights, idx, axis=0)

        # Compute squared distances
        d2 = jnp.sum((target_pos[:, None, :] - selected_points) ** 2, axis=-1)

        # Compute probability in log-space for numerical stability
        log_normalizer = jnp.log(2 * jnp.pi * self._sigma**2)
        log_phi = -0.5 * d2 / self._sigma**2
        log_weights = jnp.log(selected_weights)

        # Sum log probabilities for each target point (logsumexp for numerical stability)
        log_prob_per_point = logsumexp(log_weights + log_phi - log_normalizer, axis=1)

        # Weight by target weights and sum
        return jnp.sum(log_prob_per_point * target_weights)
