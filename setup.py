from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

import eigency

extensions = [
    Extension("NonLinearModule.NonLinearModule", ["NonLinearModule/NonLinearModule.pyx"],
        include_dirs = [".", "NonLinearModule"] + eigency.get_includes()
    ),
]

dist = setup(
    name = "NonLinearModule",
    version = "1.0",
    ext_modules = cythonize(extensions),
    packages = ["NonLinearModule"]
)

