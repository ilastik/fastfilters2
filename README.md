# fastfilters2

> SIMD-accelerated 2D and 3D image features

## Development

### Setup

The following commands use [Mambaforge][mambaforge] for installing build dependencies.
C++17 compiler and the corresponding toolchain should be available in system path.

```sh
# Create and activate a local development environment.
mamba create --name ff2 --channel conda-forge --channel ilastik-forge --yes cmake ninja python pip fastfilters pytest pytest-benchmark
mamba activate ff2

# Configure cmake.
cmake --preset main

# Install this package in editable mode.
# All changes in Python source code are picked up automatically.
pip install --editable .

# Build extension module.
# Should run this when C++ source changes.
# The module is automatically placed in the correct path.
cmake --build --preset main

# Run tests.
pytest --benchmark-skip

# Run benchmarks.
pytest --benchmark-only --benchmark-group-by=param:name,param:shape,param:scale -k benchmark_filter
```

### Add a new submodule dependency

```sh
git submodule add --name REPO_NAME REPO_URL deps/REPO_NAME
git -C deps/REPO_NAME switch --detach REPO_TAG
git add deps/REPO_NAME
git commit
```

### Update submodule dependency

```sh
git -C deps/REPO_NAME fetch
git -C deps/REPO_NAME switch --detach REPO_TAG
git add deps/REPO_NAME
git commit
```

[mambaforge]: https://github.com/conda-forge/miniforge#mambaforge
