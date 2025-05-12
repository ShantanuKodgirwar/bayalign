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

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.scipy.special import logsumexp
from pointcloud import PointCloud, RotationProjection, quat2matrix


# ---------------------------------------------------------------------- #
#  Helper: kd_nearest_k  (GPU-friendly replacement for a KD-tree)        #
# ---------------------------------------------------------------------- #
@partial(jit, static_argnames=("k",))
def _nearest_k_squared(
    y: jnp.ndarray,  # (L, dim)
    x: jnp.ndarray,  # (K, dim)
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return the k smallest squared distances from every y_l to the whole set x_k.
    Replaces the KD tree implementation on the CPU.

    Result:
        d2_out  shape (L,k)
        idx_out shape (L,k)   -- indices into x  (ties resolved arbitrarily)
    """
    # all pair-wise squared distances  (L, K)
    d2 = jnp.sum((y[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    # jax.lax.top_k returns the largest k; invert sign to get *smallest*
    d2_neg, idx = jax.lax.top_k(-d2, k)
    return -d2_neg, idx


# ---------------------------------------------------------------------- #
#  Abstract base class (tiny)                                            #
# ---------------------------------------------------------------------- #
class Registration:
    """
    Rigid registration scoring assigning a log-probability to a rigid body pose.
    Concrete subclasses implement the *SO(3)-level* score mainly via Kernel Correlation
    and Log Mixture of Gaussians.
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
    beta: float = 1.0

    def __post_init__(self):
        sig = float(self.sigma)
        bt = float(self.beta)

        @jit
        def _logp(q):
            R = quat2matrix(q)
            x_src = self.source.positions @ R.T  # (K, dim)
            d2 = jnp.sum(
                (self.target.positions[:, None, :] - x_src[None, :, :]) ** 2, axis=-1
            )  # (L, K)
            log_w = (jnp.log(self.target.weights))[:, None] + (
                jnp.log(self.source.weights)
            )[None, :]
            val = logsumexp(-0.5 * d2 / sig**2 + log_w)
            return bt * val

        _grad = jit(grad(_logp))

        # store compiled functions
        object.__setattr__(self, "_logp", _logp)
        object.__setattr__(self, "_gradq", _grad)

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
    target: PointCloud  # 2-D
    source: RotationProjection  # 3-D → 2-D
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

        # bind static arrays once for the JIT
        tgt_pos = self.target.positions
        tgt_w = self.target.weights
        src_pos3d = self.source.positions
        src_w = self.source.weights

        @partial(jit, static_argnames=("k",))
        def _logp(q, *, k):
            """scalar log p(Y|q)  in quaternion space."""
            R = quat2matrix(q)
            src_proj = src_pos3d @ R[:-1, :].T  # (K,2)
            d2_near, idx = _nearest_k_squared(tgt_pos, src_proj, k)
            phi = jnp.exp(-0.5 * d2_near / sig**2)
            w_k = src_w[idx]
            ll = logsumexp(
                jnp.log(w_k) + jnp.log(phi) - jnp.log(2 * jnp.pi * sig**2), axis=1
            )
            return bt * jnp.sum(ll * tgt_w)

        _grad = jit(grad(_logp, argnums=0), static_argnums=("k",))

        object.__setattr__(self, "_logp", partial(_logp, k=k))
        object.__setattr__(self, "_gradq", partial(_grad, k=k))

    # public API
    def log_prob(self, q):
        return self._logp(q)

    def gradient(self, q):
        return self._gradq(q)
