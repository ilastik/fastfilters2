from setuptools import find_packages
from skbuild import setup


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_languages=("CXX",),
    cmake_minimum_required_version="3.19",
    cmake_install_dir="src/fastfilters2",
    include_package_data=True,
)
