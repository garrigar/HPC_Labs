import os
from os.path import join as pjoin
from setuptools import Extension, setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted from http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """
    Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()
print(CUDA)

# ---------------------------------------

ext = Extension('vectorsum_methods',
                sources=['sourcecode_vectorsum_methods.pyx'],
                libraries=['lib/vectorsum', 'cudart'], #, 'cublas'],
                language='c++',
                include_dirs=[CUDA['include']],
                library_dirs=[CUDA['lib64']],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                )

setup(
    include_dirs=[CUDA['include']],
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext},
)
