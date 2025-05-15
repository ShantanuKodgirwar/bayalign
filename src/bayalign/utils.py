# mainly consists of some utility functions

import contextlib
import time

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def rotation2d(angle, degree=False):
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


def format_time(t):
    units = [(1.0, "s"), (1e-3, "ms"), (1e-6, "us"), (1e-9, "ns")]

    scale, unit = None, None
    for scale, unit in units:
        if t > scale or t == 0:
            break

    return "{0:.1f} {1}".format(t / scale, unit)


@contextlib.contextmanager
def take_time(desc):
    # Synchronize GPU/TPU if using accelerators
    if jax.devices()[0].platform != "cpu":
        jax.devices()[0].synchronize_all_activity()

    t0 = time.time()
    yield

    # Synchronize again to ensure all operations are complete
    if jax.devices()[0].platform != "cpu":
        jax.devices()[0].synchronize_all_activity()

    dt = time.time() - t0
    print(f"{desc} took {format_time(dt)}")
