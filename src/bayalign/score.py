# score.py  ───────────────────────────────────────────────────────────────
"""
Rotation-dependent scores for 3D-2D rigid registration log densities implemented in JAX.

All classes implement the mini-interface required by the MCMC samplers for inference:

    .log_prob(q)      > float
    .gradient(q)      > jnp.ndarray shape (4,)   (∂/∂ quaternion)

`q` is always a **unit quaternion (x, y, z, w)**.

The module depends only on:
    * JAX (jax, jax.numpy)
    * pointcloud.PointCloud / RotationProjection helpers
"""

import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.special import logsumexp
from .pointcloud import PointCloud, quat2matrix


# ---------------------------------------------------------------------- #
#  Helper: kd_nearest_k  (GPU-friendly replacement for a KD-tree)        #
# ---------------------------------------------------------------------- #
@partial(jit, static_argnames=("k",))
def _nearest_k_squared(
    y: jnp.ndarray,  # (L, dim)
    x: jnp.ndarray,  # (K, dim)
    k: int,
    pc_threshold: int = 5000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return the k smallest squared distances from every y_l to the whole set x_k.
    Suitable for smaller point clouds as it is a Brute force method. For larger point clouds,
    or big speeds-up, check `pykeops` or the facebook package `faiss`.

    Result:
        d2_out  shape (L,k)
        idx_out shape (L,k)   -- indices into x  (ties resolved arbitrarily)
    """
    # all pair-wise squared distances  (L, K)
    d2 = jnp.sum((y[:, None, :] - x[None, :, :]) ** 2, axis=-1)

    # jax.lax.top_k returns the largest k; invert sign to get *smallest*
    d2_neg, idx = jax.lax.top_k(-d2, k)

    if y.shape[0] > pc_threshold or x.shape[0] > pc_threshold:
        warnings.warn("Large input size may impact performance")

    return -d2_neg, idx


# ---------------------------------------------------------------------- #
#  Abstract `Registration` class                                         #
# ---------------------------------------------------------------------- #
class Registration:
    """
    Rigid registration scoring assigning a log-probability to a rigid body pose.
    Concrete subclasses implement the *SO(3)-level* score mainly via Kernel Correlation
    and Log Mixture of Gaussians. In general, the score is estimating log likelihood i.e.
    log p(data | pose). All methods take a unit quaternion as input.
    """

    beta: float = 1.0  # per-metric inverse temperature

    # public API ----------------------------------------------------------
    def log_prob(self, q: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def gradient(self, q: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
#  Kernel correlation                                                    #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class KernelCorrelation(Registration):
    target: PointCloud
    source: PointCloud
    sigma: float = 1.0
    k: int = 20
    beta: float = 1.0

    def __post_init__(self):
        sig = float(self.sigma)
        k = int(self.k)
        bt = float(self.beta)

        @partial(jit, static_argnames=("k",))
        def _logp(q, *, k):
            R = quat2matrix(q)
            src_transformed = self.source.transform_positions(R)

            # nearest neighbours
            d2_near, idx = _nearest_k_squared(self.target.positions, src_transformed, k)

            log_w = (jnp.log(self.target.weights))[:, None] + (
                jnp.log(self.source.weights)
            )[None, :]
            val = logsumexp(-0.5 * d2_near / sig**2 + log_w)
            return bt * val

        _grad = jit(grad(_logp, argnums=0), static_argnums=("k",))

        # store compiled functions
        object.__setattr__(self, "_logp", partial(_logp, k=k))
        object.__setattr__(self, "_gradq", partial(_grad, k=k))

    # public API
    def log_prob(self, q):
        return self._logp(q)

    def gradient(self, q):
        return self._gradq(q)


# ---------------------------------------------------------------------- #
#  Mixture of spherical Gaussians (soft-ICP)                             #
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class MixtureSphericalGaussians(Registration):
    target: PointCloud
    source: PointCloud
    sigma: float = 1.0
    k: int = 20
    beta: float = 1.0
    """
    Log-likelihood of 2-D points under a projected 3-D Gaussian mixture.
    """

    def __post_init__(self):
        sig = float(self.sigma)
        k = int(self.k)
        bt = float(self.beta)

        @partial(jit, static_argnames=("k",))
        def _logp(q, *, k):
            """scalar log p(Y|q)  in quaternion space."""
            R = quat2matrix(q)

            # rotate and project the source
            src_transformed = self.source.transform_positions(R)  # (K,ndim)

            # nearest neighbors
            d2_near, idx = _nearest_k_squared(self.target.positions, src_transformed, k)
            phi = jnp.exp(-0.5 * d2_near / sig**2)
            w_k = self.source.weights[idx]
            ll = logsumexp(
                jnp.log(w_k) + jnp.log(phi) - jnp.log(2 * jnp.pi * sig**2), axis=1
            )
            return bt * jnp.sum(ll * self.target.weights)

        _grad = jit(grad(_logp, argnums=0), static_argnums=("k",))

        object.__setattr__(self, "_logp", partial(_logp, k=k))
        object.__setattr__(self, "_gradq", partial(_grad, k=k))

    # public API
    def log_prob(self, q):
        return self._logp(q)

    def gradient(self, q):
        return self._gradq(q)
