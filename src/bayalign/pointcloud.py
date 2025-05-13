# pointcloud_jax.py

from dataclasses import dataclass

import jax.numpy as jnp
from jaxlie import SO3


def quat2matrix(q):
    return SO3.from_quaternion_xyzw(q).as_matrix()


def matrix2quat(R):
    return SO3.from_matrix(R).quat_xyzw()


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
