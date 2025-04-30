# fastfilters2

> SIMD-accelerated 2D and 3D image features

## Development

### Prerequisites

You need the following tools installed in your system:
- C++17 compiler toolchain
- Python
- CMake

Whereas installing compiler is highly platform-specific, all other dependencies could be
obtained via [conda][conda], [mamba][mamba], or [micromamba][micromamba], which are
largely compatible with each other.

The following instructions assume that you use micromamba.

1. Create a new environment:

    ```sh
    micromamba create --yes --name fastfilters2 --strict-channel-priority --channel conda-forge --channel ilastik-forge python~=3.9.0 fastfilters
    ```

2. Activate the environment (remember to activate it every time in a new shell):

    ```sh
    micromamba activate fastfilters2
    ```

3. Install development dependencies:

    ```sh
    pip install scikit-build-core nanobind pytest imageio clang-format ruff typer rich
    ```

4. Install this package in editable mode:

    ```sh
    pip install --no-build-isolation --verbose --editable .
    ```

[conda]: https://docs.conda.io/en/latest/
[mamba]: https://mamba.readthedocs.io/en/latest/
[micromamba]: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
