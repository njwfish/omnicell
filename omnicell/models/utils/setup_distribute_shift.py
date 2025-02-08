from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import platform

# Determine platform-specific compiler and linker flags
if platform.system() == "Darwin":  # macOS
    extra_compile_args = ['-O3', '-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
    libraries = ['omp']
else:  # Linux and others
    extra_compile_args = ['-O3', '-fopenmp']
    extra_link_args = ['-fopenmp']
    libraries = ['gomp', 'pthread']  # Added pthread

ext = Extension(
    name="distribute_shift",
    sources=["distribute_shift.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries
)

setup(
    ext_modules=cythonize([ext], 
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
        }
    )
)