from typing import Literal

import numpy
import pytest

import fastfilters2.compat

fastfilters = pytest.importorskip("fastfilters")

RNG = numpy.random.Generator(numpy.random.PCG64DXSM(seed=42))
FILTERS = (
    "gaussianSmoothing",
    "gaussianGradientMagnitude",
    "laplacianOfGaussian",
    "hessianOfGaussianEigenvalues",
    "structureTensorEigenvalues",
)
SHAPES = ((512, 512), (64, 64, 64))
SCALES = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)


def assert_similar(
    actual, desired, *, mode: Literal["max", "mean"] = "max", atol: float = 1e-6
):
    __tracebackhide__ = True

    assert actual.shape == desired.shape
    assert actual.dtype == desired.dtype
    assert numpy.all(numpy.isfinite(actual))
    assert numpy.all(numpy.isfinite(desired))

    abs_diff = numpy.abs(actual - desired)

    if mode == "max":
        max_diff = numpy.max(abs_diff)
        if max_diff > atol:
            raise AssertionError(f"max(diff) = {max_diff:.2e} > {atol:.2e}")

    elif mode == "mean":
        mean_diff = numpy.mean(abs_diff)
        if mean_diff > atol:
            raise AssertionError(f"mean(diff) = {mean_diff:.2e} > {atol:.2e}")

    else:
        raise ValueError(f"Invalid mode: {mode}")


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name", FILTERS)
def test_compat(name, shape, scale):
    args = (RNG.random(shape, numpy.float32), scale)
    if name == "structureTensorEigenvalues":
        args = (*args, 0.5 * scale)

    actual = getattr(fastfilters2.compat, name)(*args)
    desired = getattr(fastfilters, name)(*args)

    mode = "mean" if len(shape) == 3 and name.endswith("Eigenvalues") else "max"
    assert_similar(actual, desired, mode=mode)
