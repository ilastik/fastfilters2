import numpy


def _is_shape(value):
    """Return True if the value is a tuple of integers."""
    return isinstance(value, tuple) and all(map(int.__instancecheck__, value))


def pytest_make_parametrize_id(config, val, argname):
    # Make names of parametrized tests prettier for certain arguments.
    if argname.endswith("shape") and _is_shape(val):
        return "x".join(map(str, val))
    if argname.endswith("dtype") and isinstance(val, numpy.dtype):
        return val.name
    return None
