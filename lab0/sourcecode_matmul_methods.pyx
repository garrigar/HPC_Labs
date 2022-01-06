cimport cython
from cython.parallel cimport prange, parallel


# defining a function from a lib
cdef extern from "cuda/matmul.h":
    float gpu_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n)

# cython wrapper for multiplying function
def cuda_matmul(float[:, :] A, float[:, :] B, float[:, :] C) -> float:
    cdef :
        int m = A.shape[0]
        int k = A.shape[1]
        int n = B.shape[1]
    return gpu_matmul(&A[0, 0], &B[0, 0], &C[0, 0], m, k, n)

# Disable all checks in order not to affect performance
@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_matmul(float[:, :] A, float[:, :] B, float[:, :] C) -> None:
    cdef:
        int i, j, k
        float res
    for i in range(A.shape[0]): # up to m
        for j in range(B.shape[1]): # up to n
            res = 0.0
            for k in range(A.shape[1]): # up to k
                res = res + A[i, k] * B[k, j]
            C[i, j] = res


@cython.overflowcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def cpu_openmp_matmul(float[:, :] A, float[:, :] B, float[:, :] C, int n_threads=8) -> None:
    cdef:
        int i, j, k
        float res
    # parallel and prange use OpenMP
    with nogil, parallel(num_threads=n_threads):
        for i in prange(A.shape[0], schedule='static'):
            for j in range(B.shape[1]):
                res = 0.0
                for k in range(A.shape[1]):
                    res = res + A[i, k] * B[k, j]
                C[i, j] = res
