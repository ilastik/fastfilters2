"""3-way comparison of results between Vigra, fastfilters, and fastfilters2."""

import fastfilters
import numpy
import vigra.filters
from rich.live import Live
from rich.table import Table

import fastfilters2.compat

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


def max_abs_diff(a, b):
    return numpy.max(numpy.abs(a - b))


def mean_square_diff(a, b):
    return numpy.mean(numpy.square(a - b))


def main():
    table = Table()
    table.add_column("filter")
    table.add_column("shape")
    table.add_column("scale", justify="right")
    table.add_column("max |vigra-ff1|", justify="right")
    table.add_column("max |vigra-ff2|", justify="right")
    table.add_column("max |ff1-ff2|", justify="right")
    table.add_column("mean (vigra-ff1)\u00b2", justify="right")
    table.add_column("mean (vigra-ff2)\u00b2", justify="right")
    table.add_column("mean (ff1-ff2)\u00b2", justify="right")

    with Live(table, auto_refresh=False) as live:
        for name in FILTERS:
            func_vigra = getattr(vigra.filters, name)
            func_ff1 = getattr(fastfilters, name)
            func_ff2 = getattr(fastfilters2.compat, name)

            for shape in SHAPES:
                data = RNG.integers(0, 256, size=shape).astype(numpy.float32)

                for scale in SCALES:
                    if name == "structureTensorEigenvalues":
                        args = (data, scale, 0.5 * scale)
                    else:
                        args = (data, scale)

                    res_vigra = func_vigra(*args)
                    res_ff1 = func_ff1(*args)
                    res_ff2 = func_ff2(*args)

                    mad_vigra_ff1 = max_abs_diff(res_vigra, res_ff1)
                    mad_vigra_ff2 = max_abs_diff(res_vigra, res_ff2)
                    mad_ff1_ff2 = max_abs_diff(res_ff1, res_ff2)

                    msd_vigra_ff1 = mean_square_diff(res_vigra, res_ff1)
                    msd_vigra_ff2 = mean_square_diff(res_vigra, res_ff2)
                    msd_ff1_ff2 = mean_square_diff(res_ff1, res_ff2)

                    style = ""
                    if mad_vigra_ff1 < mad_ff1_ff2 or msd_vigra_ff1 < msd_ff1_ff2:
                        style = "red"

                    table.add_row(
                        name,
                        "\u00d7".join(map(str, shape)),
                        f"{scale:.1f}",
                        f"{mad_vigra_ff1:.1e}",
                        f"{mad_vigra_ff2:.1e}",
                        f"{mad_ff1_ff2:.1e}",
                        f"{msd_vigra_ff1:.1e}",
                        f"{msd_vigra_ff2:.1e}",
                        f"{msd_ff1_ff2:.1e}",
                        style=style,
                    )
                    live.refresh()

            table.add_section()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
