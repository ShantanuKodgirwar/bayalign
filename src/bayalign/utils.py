# mainly consists of some utility functions

import contextlib
import time
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Scalar

from .types import Quaternion, RotationMatrix2D, RotationMatrix3D


def rotation2d(angle: Union[float, Scalar], degree: bool = False) -> RotationMatrix2D:
    if degree:
        angle = jnp.deg2rad(angle)
    return jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )


def rotation3d(euler_angles, degree=False):
    euler_angles = jnp.asarray(euler_angles)
    if degree:
        euler_angles = jnp.deg2rad(euler_angles)

    alpha, beta, gamma = euler_angles

    r_alpha = jnp.array(
        [
            [jnp.cos(alpha), -jnp.sin(alpha), 0],
            [jnp.sin(alpha), jnp.cos(alpha), 0],
            [0, 0, 1],
        ]
    )

    r_beta = jnp.array(
        [
            [jnp.cos(beta), 0, jnp.sin(beta)],
            [0, 1, 0],
            [-jnp.sin(beta), 0, jnp.cos(beta)],
        ]
    )

    r_gamma = jnp.array(
        [
            [jnp.cos(gamma), -jnp.sin(gamma), 0],
            [jnp.sin(gamma), jnp.cos(gamma), 0],
            [0, 0, 1.0],
        ]
    )

    return r_alpha @ r_beta @ r_gamma


def create_rotation(angle, degree=False):
    angle = jnp.asarray(angle)

    if angle.size == 1:
        return rotation2d(angle, degree)
    elif angle.size == 3:
        return rotation3d(angle, degree)
    else:
        raise ValueError("Angle vector should be of size 1 or 3")


def quat2matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    q : array_like
        A 4-element quaternion (x, y, z, w). Scalar last convention.

    Returns
    -------
    R : array_like
        A 3x3 rotation matrix
    """
    return Rotation.from_quat(q).as_matrix()


def matrix2quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion. Scalar last convention.

    Parameters
    ----------
    R : array_like
        A 3x3 rotation matrix.

    Returns
    -------
    array_like
        A 4-element quaternion (x, y, z, w).
    """
    return Rotation.from_matrix(R).as_quat()


class Timer:
    """Context manager for timing with JAX synchronization."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed: Optional[float] = None

    def __enter__(self):
        # Synchronize JAX devices
        if jax.devices()[0].platform != "cpu":
            jax.block_until_ready(jax.numpy.array(0))

        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Synchronize again
        if jax.devices()[0].platform != "cpu":
            jax.block_until_ready(jax.numpy.array(0))

        self.elapsed = time.perf_counter() - self.start

        if self.verbose:
            print(f"{self.name} took {format_time(self.elapsed)}")


def format_time(seconds: float) -> str:
    """Format time with appropriate units."""
    units = [(1.0, "s"), (1e-3, "ms"), (1e-6, "μs"), (1e-9, "ns")]

    for threshold, unit in units:
        if seconds >= threshold or seconds == 0:
            return f"{seconds / threshold:.1f} {unit}"

    return f"{seconds:.2e} s"


# Backward compatibility
@contextlib.contextmanager
def take_time(desc: str):
    """Legacy timer function for backward compatibility."""
    with Timer(desc) as timer:
        yield timer


def validate_rotation(
    rotation: Union[Quaternion, RotationMatrix3D],
) -> Union[Quaternion, RotationMatrix3D]:
    """
    Validate that a rotation is valid (unit quaternion or orthogonal matrix).

    Parameters
    ----------
    rotation : Quaternion or RotationMatrix3D
        Rotation to validate

    Returns
    -------
    Union[Quaternion, RotationMatrix3D]
        The same rotation if valid

    Raises
    ------
    ValueError
        If rotation is invalid
    """
    if rotation.shape == (4,):
        # Quaternion
        norm = jnp.linalg.norm(rotation)
        if not jnp.allclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"Quaternion not normalized: norm={norm}")
    elif rotation.shape == (3, 3):
        # Rotation matrix
        should_be_I = rotation @ rotation.T
        if not jnp.allclose(should_be_I, jnp.eye(3), atol=1e-6):
            raise ValueError("Matrix is not orthogonal")
        if not jnp.allclose(jnp.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(f"Matrix determinant is {jnp.linalg.det(rotation)}, not 1")
    else:
        raise ValueError(f"Invalid rotation shape: {rotation.shape}")

    return rotation
