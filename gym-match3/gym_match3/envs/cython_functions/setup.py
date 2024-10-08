from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension("cfunctions", ["cfunctions.pyx"],
              include_dirs=[numpy.get_include()],  # This tells the compiler where to find NumPy headers
              define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')])
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)
