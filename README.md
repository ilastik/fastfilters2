# fastfilters2

> SIMD-accelerated 2D and 3D image features

## Local development

The following commands use [micromamba][mamba], but `conda` is also possible. You should
also have a [C++17][c++17] compiler toolchain installed in your system.

1. Create a new development environment:

    ```sh
    micromamba create -y -n fastfilters2 -c conda-forge -c ilastik-forge python=3.9 fastfilters
    ```

2. Activate the created environment (remember to activate it every time in a new shell):

    ```sh
    micromamba activate fastfilters2
    ```

3. Install development dependencies:

    ```sh
    pip install scikit-build-core[pyproject] nanobind pytest imageio clang-format ruff typer rich
    ```

4. Install the package in editable mode:

    ```sh
    pip install --no-build-isolation -ve .
    ```

[mamba]: https://mamba.readthedocs.io
[c++17]: https://en.cppreference.com/w/cpp/17
