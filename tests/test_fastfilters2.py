import contextlib

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
COMPATIBILITY_PARAMS = {
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


def raises(exc):
    """Similar to pytest.raises, but doesn't do anything if the argument is None."""
    __tracebackhide__ = True
    return contextlib.nullcontext() if exc is None else pytest.raises(exc)


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda shape: "x".join(map(str, shape)))
@pytest.mark.parametrize("name", FILTERS)
def test_compatibility(name, shape, scale):
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

    percentile, nulp = COMPATIBILITY_PARAMS.get(name, (0, 0))
    max_ulp = numpy.spacing(vmax, dtype=dtype)
    tol = dtype(nulp) * max_ulp

    assert_allclose_trimmed(actual, desired, tol=tol, percentile=percentile)


@pytest.mark.parametrize(
    "shape, exc",
    [
        ((2, 2), None),
        ((2, 2, 2), None),
        ((2,), ValueError),
        ((2, 2, 2, 2), ValueError),
        ((0, 2), ValueError),
        ((1, 2, 2, 2), ValueError),
    ],
)
def test_shape(shape, exc):
    data = numpy.zeros(shape, dtype=numpy.float32)
    with raises(exc):
        fastfilters2.gaussian_smoothing(data, 0.3)


@pytest.mark.parametrize(
    "scale, exc",
    [(1e-5, None), (-1e-5, ValueError), (0, ValueError)],
)
def test_scale(scale, exc):
    data = numpy.zeros((2, 2), dtype=numpy.float32)
    with raises(exc):
        fastfilters2.gaussian_smoothing(data, scale)


@pytest.mark.parametrize(
    "dtype, exc",
    [
        (numpy.uint8, None),
        (numpy.float64, None),
        (numpy.dtype([("field", numpy.float32)]), TypeError),
        (numpy.void, TypeError),
    ],
)
def test_dtype(dtype, exc):
    data = numpy.zeros((2, 2), dtype=dtype)
    with raises(exc):
        out = fastfilters2.gaussian_smoothing(data, 0.3)
        assert out.dtype == numpy.float32


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2], [3, 4]],
        ((1, 2), (3, 4)),
        iter([[1, 2], [3, 4]]),
        numpy.array(["\x01\x02", "\x03\x04"]),
    ],
)
def test_reject_non_array_data(data):
    with raises(TypeError):
        fastfilters2.gaussian_smoothing(data, 0.3)


def test_non_contiguous():
    data = numpy.zeros((4, 4), dtype=numpy.float32)[::2, ::2]
    assert not data.flags.c_contiguous
    fastfilters2.gaussian_smoothing(data, 0.3)


def test_unaligned():
    raw = numpy.zeros(17, dtype=numpy.uint8)
    data = numpy.frombuffer(raw[1:], dtype=numpy.float32, count=4).reshape(2, 2)
    assert not data.flags.aligned
    fastfilters2.gaussian_smoothing(data, 0.3)
