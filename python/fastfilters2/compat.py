import functools

import numpy

from ._core import (
    gaussian_gradient_magnitude,
    gaussian_smoothing,
    hessian_of_gaussian_eigenvalues,
    laplacian_of_gaussian,
    structure_tensor_eigenvalues,
)

try:
    import vigra
except ModuleNotFoundError:
    vigra = None


def _wrap(func):
    def wrapper(array, *args, **kwargs):
        if not hasattr(array, "axistags"):
            assert all(
                d > 1 for d in array.shape
            ), "Can't handle arrays with singleton dimensions (unless they are tagged VigraArrays)."
            return func(array, *args, **kwargs)

        if vigra is None:
            raise RuntimeError("array has 'axistags', but vigra is not installed")

        squeezed = array.squeeze()
        res = func(squeezed, *args, **kwargs)
        if res.shape == squeezed.shape:
            res = vigra.taggedView(res, squeezed.axistags)
        else:
            res = vigra.taggedView(res, (vigra.AxisInfo.c, *squeezed.axistags))
        return res.withAxes(array.axistags)

    return functools.update_wrapper(wrapper, func)


@_wrap
def gaussianSmoothing(array, sigma, window_size=0.0):
    return gaussian_smoothing(array, sigma, window_ratio=window_size)


@_wrap
def gaussianGradientMagnitude(array, sigma, window_size=0.0):
    return gaussian_gradient_magnitude(array, sigma, window_ratio=window_size)


@_wrap
def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
    res = hessian_of_gaussian_eigenvalues(image, scale, window_ratio=window_size)
    return numpy.moveaxis(res, 0, -1)


@_wrap
def laplacianOfGaussian(array, scale=1.0, window_size=0.0):
    return laplacian_of_gaussian(array, scale, window_ratio=window_size)


@_wrap
def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
    res = structure_tensor_eigenvalues(
        image, innerScale, st_scale=outerScale, window_ratio=window_size
    )
    return numpy.moveaxis(res, 0, -1)
