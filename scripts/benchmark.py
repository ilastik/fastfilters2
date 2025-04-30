import timeit

import fastfilters
import numpy
from rich.live import Live
from rich.table import Table

import fastfilters2.compat


def benchmark(func, *args, **kwargs):
    timer = timeit.Timer(lambda: func(*args, **kwargs))
    iters = timer.autorange()[0]
    return min(timer.repeat(number=iters)) / iters


def benchmark_filters(names, shapes, scales):
    table = Table()
    table.add_column("Filter")
    table.add_column("Shape")
    table.add_column("Scale", justify="right")
    table.add_column("Time (old)", justify="right")
    table.add_column("Time (new)", justify="right")
    table.add_column("Speedup", justify="right")

    with Live(table, auto_refresh=False) as live:
        for name in names:
            func_old = getattr(fastfilters, name)
            func_new = getattr(fastfilters2.compat, name)

            for shape in shapes:
                data = numpy.zeros(shape, dtype=numpy.float32)

                for scale in scales:
                    args = (data, scale)
                    if name == "structureTensorEigenvalues":
                        args = (*args, 0.5 * scale)

                    time_old = benchmark(func_old, *args)
                    time_new = benchmark(func_new, *args)
                    speedup = time_old / time_new

                    if speedup < 0.9:
                        speedup_color = "red"
                    elif speedup < 1.1:
                        speedup_color = "yellow"
                    else:
                        speedup_color = "green"

                    table.add_row(
                        name,
                        "\u00d7".join(map(str, shape)),
                        f"{scale:.1f}",
                        f"{time_old * 1e3:.3f} ms",
                        f"{time_new * 1e3:.3f} ms",
                        f"[{speedup_color}]{speedup:.2f}[/{speedup_color}]",
                    )
                    live.refresh()

            table.add_section()


def main():
    # fmt: off
    filters = (
        "gaussianSmoothing",
        # "gaussianGradientMagnitude",
        # "laplacianOfGaussian",
        # "hessianOfGaussianEigenvalues",
        # "structureTensorEigenvalues",
    )
    shapes = (
        (512, 512),
        (64, 64, 64),
    )
    scales = (
        0.3,
        10.0,
    )
    # fmt: on

    benchmark_filters(filters, shapes, scales)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
