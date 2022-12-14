# fastfilters2

> SIMD-accelerated 2D and 3D image features

## Local development

The following commands use [Mambaforge][mambaforge] for installing build dependencies.
You should also have a working, modern C++ compiler toolchain installed in your system.

```sh
# Create and activate a local development environment.
mamba create --name ff2dev --yes python pip cmake ninja numpy
conda activate ff2dev

# Install this package in editable mode:
# all changes in Python source code are picked up automatically.
pip install --editable .

# Configure CMake build.
cmake --preset dev

# Build and run benchmarks: this needs to be run when C++ source files change.
cmake --build --preset dev.bm

# Build and install Python module: this needs to be run when C++ source files change.
cmake --build --preset dev.install
```

## Guides

### Add a new submodule dependency

```sh
git submodule add --name REPO https://github.com/OWNER/REPO.git deps/REPO
git -C deps/REPO switch --detach TAG
git add .
git commit --message
```

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
