from distutils.core import setup
from Cython.Build import cythonize
#cython: infer_types = True
setup(
    ext_modules = cythonize("neuralnetworkCCode.pyx")
)
