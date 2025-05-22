"""Common type definitions for bayalign using jaxtyping."""

from typing import Union

from jaxtyping import Array, Float, Int, PRNGKeyArray

# Basic array types
Float32 = Float[Array, "..."]
Int32 = Int[Array, "..."]

# Specific array types for point clouds
Points2D = Float[Array, "n 2"]
Points3D = Float[Array, "n 3"]
Points = Union[Points2D, Points3D]
Weights = Float[Array, "n"]

# Rotation representations
Quaternion = Float[Array, "4"]
RotationMatrix2D = Float[Array, "2 2"]
RotationMatrix3D = Float[Array, "3 3"]
RotationMatrix = Union[RotationMatrix2D, RotationMatrix3D]
Rotation = Union[Quaternion, RotationMatrix3D]

# Translation vectors
Translation2D = Float[Array, "2"]
Translation3D = Float[Array, "3"]
Translation = Union[Translation2D, Translation3D]

# Sampling related
SampleArray = Float[Array, "n_samples ..."]
LogProbArray = Float[Array, "n_samples"]

# Type aliases
Seed = Union[int, PRNGKeyArray]
