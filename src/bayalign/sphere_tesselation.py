import itertools

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def get_unique_rotations(rotations, decimals=10):
    """
    Remove duplicate rotation matrices by rounding and finding unique entries.

    Parameters
    ----------
    rotations : array_like
        An array of rotation matrices.
    decimals : int, optional
        Number of decimal places to round to when determining uniqueness. Default is 10.

    Returns
    -------
    array_like
        An array of unique rotation matrices.
    """
    # Get the shape of a single rotation matrix
    shape = jnp.shape(rotations)[1:]

    # Flatten each rotation matrix for comparison
    rotations = jnp.reshape(rotations, (len(rotations), -1))

    # Round the rotations and find unique matrices
    # Note: jnp.unique returns a tuple with the first element being the unique values
    unique = jnp.unique(jnp.round(rotations, decimals=decimals), axis=0)[0]

    # Reshape back to original matrix dimensions
    return jnp.reshape(unique, (len(unique),) + shape)


class C600:
    """C600

    600-Cell

    Factory class that generates a tesselation of the unit sphere in 4D
    used to cover rotation space
    """

    even_perms = (
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 2, 1, 0],
    )

    def __init__(self, upper_sphere=True):
        self.vertices = self.__class__.make_vertices(upper_sphere)

    @classmethod
    def make_vertices(cls, upper_sphere=True):
        base = [[-1, 1]] * 4
        nodes = jnp.zeros((120, 4))
        k = 2**4

        # Convert itertools.product to JAX array
        product_result = jnp.array(list(itertools.product(*base)))
        nodes = nodes.at[:k].set(product_result)

        for j in range(4):
            for s in [-2, 2]:
                nodes = nodes.at[k, j].set(s)
                k += 1

        # golden ratio
        phi = (1 + jnp.sqrt(5)) / 2

        base = base[:3]
        for perm in cls.even_perms:
            for a, b, c in itertools.product(*base):
                nodes = nodes.at[k, perm[0]].set(a * phi)
                nodes = nodes.at[k, perm[1]].set(b)
                nodes = nodes.at[k, perm[2]].set(c / phi)
                nodes = nodes.at[k, perm[3]].set(0)
                k += 1

        # normalize
        nodes = nodes * 0.5

        # keep nodes covering half sphere
        if upper_sphere:
            north = jnp.eye(4)[0]
            mask = jnp.arccos(jnp.dot(nodes, north)) <= jnp.deg2rad(120)
            nodes = nodes[mask]

        return nodes

    def create_tetrahedra(self):
        angles = jnp.round(jnp.rad2deg(jnp.arccos(self.vertices @ self.vertices.T)))
        min_val = 36.0
        mask = angles == min_val

        # Since JAX doesn't allow dynamic iteration, we'll collect all tetrahedra in a list
        tetrahedra = []
        for i_idx, j_idx, k_idx, l_idx in itertools.combinations(
            range(len(self.vertices)), 4
        ):
            if (
                mask[i_idx, j_idx]
                and mask[i_idx, k_idx]
                and mask[i_idx, l_idx]
                and mask[j_idx, k_idx]
                and mask[j_idx, l_idx]
                and mask[k_idx, l_idx]
            ):
                tetrahedra.append(
                    self.vertices[jnp.array([i_idx, j_idx, k_idx, l_idx])]
                )

        # Return all collected tetrahedra
        return tetrahedra


def split_tetrahedron(vertices):
    """Subdivide tetrahedron into eight tetrahedra and project the inner
    corners of the new tetrahedra to S3.

    Reference:
    https://www.ams.org/journals/mcom/1996-65-215/S0025-5718-96-00748-X/
    S0025-5718-96-00748-X.pdf
    """
    edge_indices = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

    def normed(x):
        return x / jnp.linalg.norm(x)

    nodes = jnp.zeros((6, 4))

    # compute nodes in center of edges
    for k, (i, j) in enumerate(edge_indices):
        nodes = nodes.at[k].set(normed(vertices[i] + vertices[j]))

    # check which skew edge is the shortest
    pairs = [(0, 5), (2, 4), (3, 1)]
    dots = jnp.array([nodes[i] @ nodes[j] for i, j in pairs])
    index = jnp.argmax(dots)

    tetrahedra = [
        (vertices[0], nodes[0], nodes[2], nodes[3]),
        (vertices[1], nodes[0], nodes[1], nodes[4]),
        (vertices[2], nodes[1], nodes[2], nodes[5]),
        (vertices[3], nodes[3], nodes[4], nodes[5]),
    ]

    # Use jax.lax.cond instead of if-else for JAX compatibility
    def index_0_branch(x):
        return [
            (nodes[0], nodes[5], nodes[3], nodes[2]),
            (nodes[0], nodes[5], nodes[4], nodes[3]),
            (nodes[5], nodes[0], nodes[1], nodes[4]),
            (nodes[5], nodes[0], nodes[1], nodes[2]),
        ]

    def index_1_branch(x):
        return [
            (nodes[0], nodes[4], nodes[3], nodes[2]),
            (nodes[0], nodes[1], nodes[4], nodes[2]),
            (nodes[5], nodes[2], nodes[1], nodes[4]),
            (nodes[5], nodes[3], nodes[2], nodes[4]),
        ]

    def index_2_branch(x):
        return [
            (nodes[3], nodes[1], nodes[0], nodes[2]),
            (nodes[3], nodes[1], nodes[4], nodes[0]),
            (nodes[3], nodes[1], nodes[5], nodes[4]),
            (nodes[1], nodes[3], nodes[2], nodes[5]),
        ]

    additional_tetrahedra = jax.lax.cond(
        index == 0,
        index_0_branch,
        lambda x: jax.lax.cond(index == 1, index_1_branch, index_2_branch, None),
        None,
    )

    tetrahedra.extend(additional_tetrahedra)

    return jnp.array(tetrahedra)


def tessellate_rotations(n_discretize=2):
    """
    Discretize tetrahedra into finer tetrahedra and returns as quaternion
    based on the degree of discretization.

    :param n_discretize: positive int
        degree of discretization
    :return: discretized_quat: ndarray
        discretized tetrahedra converted to quaternion
    """
    cell600 = C600()
    tetrahedra = jnp.array(list(cell600.create_tetrahedra()))

    # Since JAX doesn't support dynamic loops well, we'll use a fixed loop
    for i in range(n_discretize):
        # Map split_tetrahedron over all tetrahedra
        # This part is tricky without jit/vmap, so we'll use a list comprehension
        split_results = [split_tetrahedron(tet) for tet in tetrahedra]
        tetrahedra = jnp.reshape(jnp.array(split_results), (-1, 4, 4))

    discretized_quat = jnp.mean(tetrahedra, axis=1)
    discretized_quat = (
        discretized_quat / jnp.linalg.norm(discretized_quat, axis=1)[:, None]
    )

    # Convert to rotation matrix (using a list comprehension since we're not using vmap)
    rotations = jnp.array([Rotation.from_quat(q).as_matrix() for q in discretized_quat])
    rotations = get_unique_rotations(rotations)

    # Convert back to quaternion
    discretized_quat = jnp.array([Rotation.from_matrix(R).as_quat() for R in rotations])

    return discretized_quat
