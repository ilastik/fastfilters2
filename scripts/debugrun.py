"""Run a command with ASan and UBSan enabled."""

import os
import shutil
import sys
import subprocess
import platform
from typing import Union

# https://github.com/google/sanitizers/wiki/SanitizerCommonFlags
COMMON_OPTIONS = (
    "verbosity=1",
    "abort_on_error=1",
)

# https://github.com/google/sanitizers/wiki/AddressSanitizerFlags
ASAN_OPTIONS = (
    "check_initialization_order=1",
    "detect_stack_use_after_return=1",
)

UBSAN_OPTIONS = ("print_stacktrace=1",)


def path_of(cmd: str) -> str:
    """Return the path to a command, or raise an error if it is not found."""
    path = shutil.which(cmd)
    if path is None:
        raise RuntimeError(f"{cmd} not found")
    return path


def stdout_from(*cmd: str) -> str:
    """Run a command, check for errors, and return its stdout."""
    return subprocess.check_output(cmd, text=True).strip()


def path_of_lib(compiler: str, lib: str) -> str:
    """Return the path to a library, or raise an error if it is not found."""
    path = stdout_from(compiler, "--print-file-name", lib)
    if path == "":
        raise RuntimeError(f"{lib} not found")
    return path


def main(cmd: list[str]) -> Union[str, int]:
    """Run a command under ASan and UBSan."""
    system = platform.system()
    env = {
        # PyMalloc is the default Python allocator, and it doesn't play well with ASan.
        "PYTHONMALLOC": "malloc",
        "ASAN_OPTIONS": ":".join(COMMON_OPTIONS + ASAN_OPTIONS),
        "UBSAN_OPTIONS": ":".join(COMMON_OPTIONS + UBSAN_OPTIONS),
    }

    if system == "Linux":
        compiler = path_of("gcc")
        asan = path_of_lib(compiler, "libasan.so")
        ubsan = path_of_lib(compiler, "libubsan.so")
        env["LD_PRELOAD"] = f"{asan}:{ubsan}"

    elif system == "Darwin":
        compiler = path_of("clang")
        asan = path_of_lib(compiler, "libclang_rt.asan_osx_dynamic.dylib")
        ubsan = path_of_lib(compiler, "libclang_rt.ubsan_osx_dynamic.dylib")
        env["DYLD_INSERT_LIBRARIES"] = f"{asan}:{ubsan}"

    else:
        raise NotImplementedError(f"{system} is not supported")

    print("Added environment variables:")
    for k, v in env.items():
        print(f"  {k}={v}")
    print("-" * shutil.get_terminal_size().columns)

    return subprocess.run(cmd, env={**os.environ, **env}).returncode


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(f"Usage: {sys.argv[0]} <command>")
    sys.exit(main(sys.argv[1:]))
