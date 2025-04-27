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
    def wrapper(src, *args, **kwargs):
        if not hasattr(src, "axistags"):
            return func(src, *args, **kwargs)

        if vigra is None:
            raise ModuleNotFoundError(
                "can't handle Vigra arrays unless 'vigra' is installed"
            )

        data = src.squeeze()
        out = func(data, *args, **kwargs)

        axistags = list(data.axistags)
        if out.ndim > data.ndim:
            axistags.append(vigra.AxisInfo.c)

        return vigra.taggedView(out, axistags).withAxes(src.axistags)

    return functools.update_wrapper(wrapper, func)


@_wrap_vigra
def gaussianSmoothing(array, sigma, window_size=0.0):
    return fastfilters2.gaussian_smoothing(array, sigma, truncate=window_size)


@_wrap_vigra
def gaussianGradientMagnitude(array, sigma, window_size=0.0):
    return fastfilters2.gaussian_gradient_magnitude(array, sigma, truncate=window_size)


@_wrap_vigra
def hessianOfGaussianEigenvalues(image, scale, window_size=0.0):
    out = fastfilters2.hessian_of_gaussian_eigenvalues(
        image, scale, truncate=window_size
    )
    return numpy.moveaxis(out, 0, -1)


@_wrap_vigra
def laplacianOfGaussian(array, scale=1.0, window_size=0.0):
    return fastfilters2.laplacian_of_gaussian(array, scale, truncate=window_size)


@_wrap_vigra
def structureTensorEigenvalues(image, innerScale, outerScale, window_size=0.0):
    out = fastfilters2.structure_tensor_eigenvalues(
        image, innerScale, derivative_scale=outerScale, truncate=window_size
    )
    return numpy.moveaxis(out, 0, -1)


@_wrap_vigra
def gaussianDerivative(array, sigma, order, window_size=0.0):
    if isinstance(order, list):
        assert len(order) == len(array.shape)
        assert len(set(order)) == 1
        order = order[0]
    return fastfilters2.gaussian_derivative(
        array, sigma, order=order, truncate=window_size
    )
