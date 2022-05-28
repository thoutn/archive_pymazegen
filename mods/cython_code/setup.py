from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.error_on_unknown_names = False

setup(
    ext_modules=cythonize("maze_nng_cy.pyx", annotate=True)
)