import numpy
import pytest

import fastfilters2
import fastfilters2.compat

fastfilters = pytest.importorskip("fastfilters")

RNG = numpy.random.default_rng(seed=42)
FILTERS = (
    "gaussianSmoothing",
    "gaussianGradientMagnitude",
    "laplacianOfGaussian",
    "hessianOfGaussianEigenvalues",
    "structureTensorEigenvalues",
)
SHAPES = ((512, 512), (64, 64, 64))
SCALES = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

# For each particular filter, the top percentile and the number of ULPs to allow for the
# compatibility test. Unfortunately, it seems impossible to achieve the bit-for-bit
# identical results without the loss of performance.
COMPAT_PARAMS = {
    "gaussianGradientMagnitude": (0, 1),
    "laplacianOfGaussian": (0, 2),
    "hessianOfGaussianEigenvalues": (0.01, 20),
    "structureTensorEigenvalues": (0.01, 60),
}


def assert_allclose_trimmed(a, b, *, tol, percentile=0):
    """Assert that two arrays are close enough, but optionally allow for outliers.

    The arrays are close if the given top percentile of the absolute differences between
    the arrays' elements is no more than `tol`.

    If the percentile is too small or 0, take the maximum absolute difference.
    """
    __tracebackhide__ = True
    k = numpy.clip(round(0.01 * percentile * a.size), 1, a.size)
    abs_diff = numpy.partition(numpy.abs(a - b), -k, axis=None)[-k]
    if abs_diff > tol:
        raise AssertionError(f"abs_diff = {abs_diff:.2e} > {tol:.2e} (k = {k})")


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", FILTERS)
def test_compat(name, shape, scale):
    dtype = numpy.float32
    vmin, vmax = 0, 255
    data = RNG.integers(vmin, vmax, size=shape, endpoint=True).astype(dtype)

    if name == "structureTensorEigenvalues":
        args = (data, scale, 0.5 * scale)
    else:
        args = (data, scale)

    func_ff2 = getattr(fastfilters2.compat, name)
    func_ff1 = getattr(fastfilters, name)

    actual = func_ff2(*args)
    desired = func_ff1(*args)

    percentile, nulp = COMPAT_PARAMS.get(name, (0, 0))
    max_ulp = numpy.spacing(vmax, dtype=dtype)
    tol = dtype(nulp) * max_ulp

    assert_allclose_trimmed(actual, desired, tol=tol, percentile=percentile)


def test_compat_small_data():
    data = numpy.arange(24).reshape(2, 3, 4).astype(numpy.float32)
    scale = 0.3
    actual = fastfilters2.gaussian_smoothing(data, scale)
    desired = fastfilters.gaussianSmoothing(data, scale)
    tol = numpy.spacing(numpy.max(data), dtype=numpy.float32)
    assert_allclose_trimmed(actual, desired, tol=tol)
