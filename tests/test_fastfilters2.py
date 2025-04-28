import contextlib

import numpy
import pytest

import fastfilters2


def raises(exc):
    """Similar to pytest.raises, but doesn't do anything if the argument is None."""
    __tracebackhide__ = True
    return contextlib.nullcontext() if exc is None else pytest.raises(exc)


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
    ids=[
        "nested_list",
        "nested_tuple",
        "iterable",
        "string_array",
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
