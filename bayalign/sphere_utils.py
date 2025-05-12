"""
Some utility functions for sampling from the sphere using JAX.
"""

import jax.numpy as jnp
from jax import random

# projections


def radial_projection(x):
    """
    Radial projection of (d+1)-dimensional point(s) to d-sphere.
    """
    norm = jnp.linalg.norm(x, axis=-1) + 1e-100
    if x.ndim == 1:
        return x / norm
    else:
        return x / norm[:, None]


def orthogonal_projection(x, y):
    """
    Map point(s) `x` into orthogonal complement of point `y`.
    """
    normal = radial_projection(y)
    return x - (x @ normal) * normal


def spherical_projection(x, v):
    """
    Map point(s) `x` into the great subsphere with pole `v`.
    """
    return radial_projection(orthogonal_projection(x, v))


# sampling


def sample_sphere(key, d=2, size=None):
    """
    Draw a random point from d-sphere by drawing a (d+1)-dimensional point from the
    standard normal distribution and mapping it to d-sphere.
    """
    if size is None:
        x = random.normal(key, shape=(d + 1,))
    else:
        x = random.normal(key, shape=(size, d + 1))
    return radial_projection(x)


def sample_subsphere(key, v):
    """
    Sample uniformly from the great subsphere with pole `v`.
    """
    return spherical_projection(random.normal(key, shape=(len(v),)), v)


# distances


def distance(x, y):
    """
    Great circle distance (assuming x, y are on the sphere).
    """
    return jnp.arccos(jnp.clip(jnp.sum(x * y, axis=-1), -1, 1))


# transformation between Cartesian and polar coordinates


def cartesian2polar(x):
    """Map points on the unit sphere to polar coordinates."""
    return jnp.mod(jnp.arctan2(x[:, 1], x[:, 0]), 2 * jnp.pi)


def polar2cartesian(theta):
    return jnp.transpose(jnp.stack([jnp.cos(theta), jnp.sin(theta)]))


def spherical2cartesian(phi, theta):
    return jnp.transpose(
        jnp.stack([
            jnp.cos(phi) * jnp.sin(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(theta)
        ])
    )


def cartesian2spherical(x):
    return cartesian2polar(x), jnp.mod(jnp.arccos(x[:, 2]), jnp.pi)


def sample_marginal(key, d, size=None):
    # JAX implementation of beta sampling
    s = random.beta(key, 0.5, 0.5 * (d - 1), shape=() if size is None else (size,))
    sign_key = random.split(key)[1]
    t = jnp.sqrt(s) * jnp.where(
        random.uniform(sign_key, shape=() if size is None else (size,)) < 0.5,
        -1.0,
        1.0
    )
    return t


def wrap(x, u, v):
    """See Mardia and Jupp for 'wrapping approach'."""
    theta = jnp.linalg.norm(x)
    return jnp.sin(theta) * u + jnp.cos(theta) * v


def slerp(u, v):
    theta = jnp.arccos(u @ v)

    def interpolation(phi):
        return (jnp.sin(phi) * v + jnp.sin(theta - phi) * u) / jnp.sin(theta)

    return interpolation


def givens(u, v, x):
    """
    Returns function that rotates `x` in the plane spanned by `u`, `v` by an
    angle that will be the argument of the function.
    """

    def rotate(theta):
        ux = u @ x
        vx = v @ x
        return (
            x
            + (jnp.cos(theta) - 1) * (ux * u + vx * v)
            + jnp.sin(theta) * (ux * v - vx * u)
        )

    return rotate
