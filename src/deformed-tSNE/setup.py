import numpy as np

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
  name = "deformed_tSNE",
  version = "0.1.0",
  ext_package = "deformed_tSNE",
  ext_modules = cythonize([
    Extension('utils', ['deformed_tSNE/utils.pyx']),
    Extension('barnes_hut', ['deformed_tSNE/barnes_hut.pyx'])]),
  include_dirs = [np.get_include()])
