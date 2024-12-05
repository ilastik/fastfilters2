import functools

import numpy

import fastfilters2

try:
    import vigra
except ModuleNotFoundError:
    vigra = None

__all__ = (
    "gaussianSmoothing",
    "gaussianGradientMagnitude",
    "hessianOfGaussianEigenvalues",
    "laplacianOfGaussian",
    "structureTensorEigenvalues",
    "gaussianDerivative",
)


def _wrap_vigra(func):
    def wrapper(array, *args, **kwargs):
        if not hasattr(array, "axistags"):
            return func(array, *args, **kwargs)

        if vigra is None:
            raise ModuleNotFoundError(
                "can't handle Vigra arrays unless 'vigra' library is installed"
            )

        array = vigra.taggedView(numpy.ascontiguousarray(array), array.axistags)
        squeezed = array.squeeze()
        res = func(squeezed, *args, **kwargs)

        if res.shape == squeezed.shape:
            res = vigra.taggedView(res, squeezed.axistags)
        else:
            res = vigra.taggedView(res, [*squeezed.axistags, vigra.AxisInfo.c])

        return res.withAxes(array.axistags)

    return functools.update_wrapper(wrapper, func)


@_wrap_vigra
def gaussianSmoothing(array, sigma, window_size=0.0):
    return fastfilters2.gaussian_smoothing(array, sigma, truncate=window_size)


@_wrap_vigra
def gaussianGradientMagnitude(array, sigma, window_size=0.0):
    return fastfilters2.gaussian_gradient_magnitude(array, sigma, truncate=window_size)


@_wrap_vigra
def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
    res = fastfilters2.hessian_of_gaussian_eigenvalues(
        image, scale, truncate=window_size
    )
    return numpy.moveaxis(res, 0, -1)


@_wrap_vigra
def laplacianOfGaussian(array, scale=1.0, window_size=0.0):
    return fastfilters2.laplacian_of_gaussian(array, scale, truncate=window_size)


@_wrap_vigra
def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
    res = fastfilters2.structure_tensor_eigenvalues(
        image, outerScale, truncate=window_size, smooth_scale=innerScale
    )
    return numpy.moveaxis(res, 0, -1)


@_wrap_vigra
def gaussianDerivative(array, sigma, order, window_size=0.0):
    if isinstance(order, list):
        assert len(order) == len(array.shape)
        assert len(set(order)) == 1
        order = order[0]
    return fastfilters2.gaussian_smoothing(
        array, sigma, truncate=window_size, order=order
    )
