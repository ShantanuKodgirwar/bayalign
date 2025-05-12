# pointcloud_jax.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _quat_normalise(q: jnp.ndarray) -> jnp.ndarray:
    """Return q / ‖q‖ with safe NaNs if q≈0 ."""
    return q / jnp.linalg.norm(q)


def quat2matrix(q: jnp.ndarray) -> jnp.ndarray:
    """
    Quaternion (x,y,z,w) → 3×3 rotation matrix.
    Same formula as SciPy, now with jax.numpy so it is jit- and grad-able.
    """
    q = _quat_normalise(q)
    x, y, z, w = q

    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix2quat(R: jnp.ndarray) -> jnp.ndarray:
    """3×3 rotation matrix → quaternion (x,y,z,w)."""
    # closed-form adapted for JAX (no SciPy).  Works for proper rotations.
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    def _branch1():
        s = jnp.sqrt(trace + 1.0) * 2.0
        return jnp.array([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s])

    def _branch2():
        def _i_max0():
            s = jnp.sqrt(1.0 + m00 - m11 - m22) * 2.0
            return jnp.array(
                [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]
            )

        def _i_max1():
            s = jnp.sqrt(1.0 + m11 - m00 - m22) * 2.0
            return jnp.array(
                [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]
            )

        s = jnp.sqrt(1.0 + m22 - m00 - m11) * 2.0
        return jnp.array([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s])

    return jax.lax.cond(trace > 0.0, _branch1, _branch2)


@dataclass(frozen=True)
class PointCloud:
    """
    Immutable, JIT-friendly weighted point cloud in 2-D or 3-D.
    All arrays are JAX DeviceArrays.
    """

    positions: jnp.ndarray  # shape (N,2) or (N,3)
    weights: jnp.ndarray  # shape (N,)

    def __post_init__(self):
        object.__setattr__(
            self, "positions", jnp.asarray(self.positions, dtype=jnp.float32)
        )
        if self.weights is None:
            w = jnp.ones(self.positions.shape[0], dtype=self.positions.dtype)
        else:
            w = jnp.asarray(self.weights, dtype=self.positions.dtype)
        object.__setattr__(self, "weights", w)

        # sanity checks (executed once, not in jitted traces)
        if self.positions.ndim != 2 or self.positions.shape[1] not in (2, 3):
            raise ValueError("positions must be (N,2) or (N,3)")
        if self.weights.shape != (self.positions.shape[0],):
            raise ValueError("weights must be 1-D array of length N")

    # ------------- convenient properties ---------------------------------
    @property
    def dim(self) -> int:
        return int(self.positions.shape[1])

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])

    @property
    def center_of_mass(self) -> jnp.ndarray:
        return jnp.sum(self.positions * self.weights[:, None], axis=0) / jnp.sum(
            self.weights
        )

    # ------------- geometry ----------------------------------------------
    def transform_positions(
        self, rotation: jnp.ndarray, translation: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """
        Apply rotation (matrix or quaternion) and optional translation;
        returns **new** array, leaves self untouched.
        """
        R = quat2matrix(rotation) if rotation.shape == (4,) else rotation
        if translation is None:
            translation = jnp.zeros(
                (R.shape[0] - (1 if self.dim == 2 else 0),), dtype=R.dtype
            )
        return self.positions @ R.T + translation

    # For compatibility with the NumPy version,
    # provide an *immutable* replacement for .transform(…)
    def transformed(self, rotation, translation=None) -> "PointCloud":
        return PointCloud(self.transform_positions(rotation, translation), self.weights)


@dataclass(frozen=True)
class RotationProjection(PointCloud):
    """
    Rotate a 3-D cloud and project onto XY, giving a 2-D cloud.
    Inherits `positions` as (N,3) and `weights` as (N,).
    """

    def transform_positions(
        self, rotation: jnp.ndarray, translation: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        R = quat2matrix(rotation) if rotation.shape == (4,) else rotation
        if translation is None:
            translation = jnp.zeros(2, dtype=R.dtype)
        # project: keep first two rows of R before transposing
        return self.positions @ R[:-1].T + translation
