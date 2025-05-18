"""
JAX-compatible point cloud implementations for rigid registration.
Provides immutable PointCloud and RotationProjection classes that work well with JAX.
"""

from typing import ClassVar, Optional, Union

import jax
import jax.numpy as jnp

from .utils import quat2matrix


class PointCloud:
    """
    JAX-friendly weighted point cloud in 2-D or 3-D.

    This class is designed to work seamlessly with JAX transformations including JIT,
    gradients, and other JAX operations. All data is stored as JAX DeviceArrays.
    """

    # Class variables to identify the type during tracing
    _cloud_type: ClassVar[str] = "point_cloud"

    def __init__(
        self,
        positions: Union[jnp.ndarray, list, tuple],
        weights: Optional[Union[jnp.ndarray, list, tuple]] = None,
    ):
        """
        Initialize a point cloud with positions and optional weights.

        Parameters
        ----------
        positions : array-like
            Point positions with shape (N, D) where D is 2 or 3
        weights : array-like, optional
            Point weights with shape (N,). If None, uniform weights are used.
        """
        # Convert inputs to JAX arrays
        self._positions = jnp.asarray(positions, dtype=jnp.float32)

        # Create uniform weights if none provided
        if weights is None:
            self._weights = jnp.ones(
                self._positions.shape[0], dtype=self._positions.dtype
            )
        else:
            self._weights = jnp.asarray(weights, dtype=self._positions.dtype)

        # Validate dimensions
        if self._positions.ndim != 2 or self._positions.shape[1] not in (2, 3):
            raise ValueError("positions must be (N, 2) or (N, 3)")

        if self._weights.shape != (self._positions.shape[0],):
            raise ValueError("weights must be 1-D array of length N")

        # Pre-calculate center of mass for efficiency
        self._center_of_mass = jnp.sum(
            self._positions * self._weights[:, None], axis=0
        ) / jnp.sum(self._weights)

    @property
    def positions(self) -> jnp.ndarray:
        """Get the point positions."""
        return self._positions

    @property
    def weights(self) -> jnp.ndarray:
        """Get the point weights."""
        return self._weights

    @property
    def dim(self) -> int:
        """Get the dimensionality (2 or 3)."""
        return int(self._positions.shape[1])

    @property
    def size(self) -> int:
        """Get the number of points."""
        return int(self._positions.shape[0])

    @property
    def center_of_mass(self) -> jnp.ndarray:
        """Get the weighted center of mass."""
        return self._center_of_mass

    def centered_copy(self):
        """
        Create a new point cloud with positions centered at the origin.

        Returns
        -------
        PointCloud
            A new instance of the same class with positions centered at (0,0,0)
        """
        # Create new point cloud with centered positions
        # Use self.__class__ to ensure the correct subclass is instantiated
        return self.__class__(self._positions - self._center_of_mass, self._weights)

    def transform_positions(
        self, rotation: jnp.ndarray, translation: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply rotation (matrix or quaternion) and optional translation.
        Returns a new array with transformed positions.

        Parameters
        ----------
        rotation : jnp.ndarray
            Rotation matrix with shape (D, D) or quaternion with shape (4,)
        translation : jnp.ndarray, optional
            Translation vector with shape (D,)

        Returns
        -------
        jnp.ndarray
            Transformed positions with shape (N, D)
        """
        # Convert quaternion to rotation matrix if needed
        R = quat2matrix(rotation) if rotation.shape == (4,) else rotation

        # Default translation is zero
        if translation is None:
            translation = jnp.zeros((self.dim,), dtype=R.dtype)

        # Apply transformation: R * p + t
        return jnp.dot(self._positions, R.T) + translation

    def transformed(self, rotation, translation=None) -> "PointCloud":
        """
        Create a new PointCloud with transformed positions.

        Parameters
        ----------
        rotation : jnp.ndarray
            Rotation matrix or quaternion
        translation : jnp.ndarray, optional
            Translation vector

        Returns
        -------
        PointCloud
            New point cloud with transformed positions
        """
        # Get transformed positions
        transformed_positions = self.transform_positions(rotation, translation)

        # Create new point cloud with same weights
        return PointCloud(transformed_positions, self._weights)

    def tree_flatten(self):
        """
        Flatten the class for JAX transformations.
        This enables JAX pytree compatibility for the class.
        """
        # Store the arrays as children
        children = (self._positions, self._weights)

        # Store other attributes as auxiliary data
        aux_data = {}

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct the class from flattened data.
        This enables JAX pytree compatibility for the class.
        """
        positions, weights = children

        # Create a new instance directly with arrays
        instance = cls.__new__(cls)
        instance._positions = positions
        instance._weights = weights
        instance._center_of_mass = jnp.sum(
            positions * weights[:, None], axis=0
        ) / jnp.sum(weights)

        return instance


class RotationProjection(PointCloud):
    """
    Rotate a 3-D cloud and project onto XY, giving a 2-D cloud.
    Inherits from PointCloud but overrides transform_positions.
    """

    # Class variable to identify the type during tracing
    _cloud_type: ClassVar[str] = "rotation_projection"

    def __init__(
        self,
        positions: Union[jnp.ndarray, list, tuple],
        weights: Optional[Union[jnp.ndarray, list, tuple]] = None,
    ):
        """
        Initialize a 3D point cloud for 2D projection.

        Parameters
        ----------
        positions : array-like
            3D point positions with shape (N, 3)
        weights : array-like, optional
            Point weights with shape (N,). If None, uniform weights are used.
        """
        super().__init__(positions, weights)

        # Verify that input is 3D
        if self._positions.shape[1] != 3:
            raise ValueError("RotationProjection requires 3D input points (N, 3)")

    def transform_positions(
        self, rotation: jnp.ndarray, translation: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Apply 3D rotation and then project to 2D (dropping Z coordinate).
        Optionally applies a 2D translation after projection.

        Parameters
        ----------
        rotation : jnp.ndarray
            3D rotation matrix with shape (3, 3) or quaternion with shape (4,)
        translation : jnp.ndarray, optional
            2D translation vector with shape (2,)

        Returns
        -------
        jnp.ndarray
            Transformed and projected positions with shape (N, 2)
        """
        # Convert quaternion to rotation matrix if needed
        R = quat2matrix(rotation) if rotation.shape == (4,) else rotation

        # Default translation is zero in 2D
        if translation is None:
            translation = jnp.zeros(2, dtype=R.dtype)

        # Rotate in 3D and then project to 2D by using only the first 2 rows of R
        # This is equivalent to rotating and then dropping the Z coordinate
        return jnp.dot(self._positions, R[:-1, :].T) + translation

    def transformed(self, rotation, translation=None) -> PointCloud:
        """
        Create a new 2D PointCloud with transformed and projected positions.

        Parameters
        ----------
        rotation : jnp.ndarray
            3D rotation matrix or quaternion
        translation : jnp.ndarray, optional
            2D translation vector

        Returns
        -------
        PointCloud
            New 2D point cloud with transformed positions
        """
        # Get transformed positions (already 2D after projection)
        transformed_positions = self.transform_positions(rotation, translation)

        # Create new 2D point cloud
        return PointCloud(transformed_positions, self._weights)


# Register PointCloud and RotationProjection as JAX PyTrees
jax.tree_util.register_pytree_node_class(PointCloud)
jax.tree_util.register_pytree_node_class(RotationProjection)
