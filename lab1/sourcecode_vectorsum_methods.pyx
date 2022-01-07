cimport cython
# from cython.parallel cimport prange, parallel

# defining a function from a lib
cdef extern from "cuda/vectorsum.h":
    ctypedef struct timed_result:
        int result
        float time
    timed_result gpu_vectorsum(int *arr, int n)

# cython wrapper for vectorsum function
def cuda_vectorsum(int[:] vec):
    cdef:
        n = len(vec)
    return gpu_vectorsum(&vec[0], n)

# Disable all checks in order not to affect performance
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_vectorsum(int[:] vec):
    cdef:
        int i
        int res

    res = 0
    for i in range(len(vec)):
        res += vec[i]
    
    return res

# @cython.overflowcheck(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def cpu_openmp_vectorsum(int[:] vec, int n_threads=8):
#     cdef:
#         int i
#         int res

#     res = 0

#     with nogil, parallel(num_threads=n_threads):
#         for i in prange(len(vec)):
#             res += vec[i]
    
#     return res
