"""3-way comparison of results between Vigra, fastfilters, and fastfilters2."""

import itertools

import fastfilters
import numpy
import vigra.filters
from rich.live import Live
from rich.table import Table

import fastfilters2.compat

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


def abs_max_mean_diff(a, b):
    diff = numpy.abs(a - b)
    return numpy.max(diff), numpy.mean(diff)


def main():
    table = Table()

    table.add_column("filter")
    table.add_column("shape")
    table.add_column("scale", justify="right")

    table.add_column("max |ff2-ff1|", justify="right")
    table.add_column("mean |ff2-ff1|", justify="right")

    table.add_column("max |ff1-vigra|", justify="right")
    table.add_column("mean |ff1-vigra|", justify="right")

    table.add_column("max |ff2-vigra|", justify="right")
    table.add_column("mean |ff2-vigra|", justify="right")

    with Live(table, auto_refresh=False) as live:
        for name, shape, scale in itertools.product(FILTERS, SHAPES, SCALES):
            func_ff2 = getattr(fastfilters2.compat, name)
            func_ff1 = getattr(fastfilters, name)
            func_vigra = getattr(vigra.filters, name)

            data = RNG.random(shape, numpy.float32)

            if name == "structureTensorEigenvalues":
                res_ff2 = func_ff2(data, scale, 0.5 * scale)
                res_ff1 = func_ff1(data, scale, 0.5 * scale)
                # In vigra, inner and outer scales are swapped.
                res_vigra = func_vigra(data, 0.5 * scale, scale)
            else:
                res_ff2 = func_ff2(data, scale)
                res_ff1 = func_ff1(data, scale)
                res_vigra = func_vigra(data, scale)

            max_ff2_ff1, mean_ff2_ff1 = abs_max_mean_diff(res_ff2, res_ff1)
            max_ff1_vigra, mean_ff1_vigra = abs_max_mean_diff(res_ff1, res_vigra)
            max_ff2_vigra, mean_ff2_vigra = abs_max_mean_diff(res_ff2, res_vigra)

            table.add_row(
                name,
                "\u00d7".join(map(str, shape)),
                f"{scale:.1f}",
                f"{max_ff2_ff1:.2e}",
                f"{mean_ff2_ff1:.2e}",
                f"{max_ff1_vigra:.2e}",
                f"{mean_ff1_vigra:.2e}",
                f"{max_ff2_vigra:.2e}",
                f"{mean_ff2_vigra:.2e}",
            )

            live.refresh()

    table.add_section()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
