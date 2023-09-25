from typing import SupportsFloat, SupportsInt

import numpy
from numpy.typing import ArrayLike, NDArray

def gaussian_kernel(
    scale: SupportsFloat, *, order: SupportsInt = 0
) -> NDArray[numpy.float32]: ...
def gaussian_smoothing(
    data: ArrayLike, scale: SupportsFloat, *, window_ratio: SupportsFloat = 0.0
) -> NDArray[numpy.float32]: ...
def gaussian_gradient_magnitude(
    data: ArrayLike, scale: SupportsFloat, *, window_ratio: SupportsFloat = 0.0
) -> NDArray[numpy.float32]: ...
def laplacian_of_gaussian(
    data: ArrayLike, scale: SupportsFloat, *, window_ratio: SupportsFloat = 0.0
) -> NDArray[numpy.float32]: ...
def hessian_of_gaussian_eigenvalues(
    data: ArrayLike, scale: SupportsFloat, *, window_ratio: SupportsFloat = 0.0
) -> NDArray[numpy.float32]: ...
def structure_tensor_eigenvalues(
    data: ArrayLike,
    scale: SupportsFloat,
    *,
    st_scale: SupportsFloat = 0.0,
    window_ratio: SupportsFloat = 0.0,
) -> NDArray[numpy.float32]: ...
