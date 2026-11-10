import pybind11
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "neural_net",  # Must match PYBIND11_MODULE name in C++
        ["C++Library.cpp"],
        include_dirs=[pybind11.get_include()],
    ),
]

setup(
    name="neural_net",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)