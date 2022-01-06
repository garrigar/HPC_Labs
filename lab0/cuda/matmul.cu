#include <iostream>
#include <cublas_v2.h>

#define GPU_ID 0

cublasHandle_t HANDLE = NULL;
float *dev_A = NULL, *dev_B = NULL, *dev_C = NULL;
int LAST_M = -1, LAST_N = -1, LAST_K = -1;


static void handleError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", file, line);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) ( handleError( err, __FILE__, __LINE__ ) )


void gpu_blas_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda = m;
    int ldb = k;
    int ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    if (HANDLE == NULL){
        cublasCreate(&HANDLE); 
    }
    
    // C = alpha * op(A)op(B) + beta * C
    // ld? - leading dimensions of the matrices
    cublasSgemm(HANDLE, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
    
    // cublasDestroy(HANDLE); 
}

// float* cudaMallocMatrix(const int m, const int n){
//     float *dev_matrix;
//     HANDLE_ERROR(cudaMalloc(&dev_matrix, m * n * sizeof(float)));
//     return dev_matrix;
// }

// returns: time in ms
float gpu_matmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    cudaSetDevice(GPU_ID);

    size_t size_A = m * k * sizeof(float);
    size_t size_B = k * n * sizeof(float);
    size_t size_C = m * n * sizeof(float);

    // reallocating memory only if new dims are different
    if ((m != LAST_M) || (k != LAST_K) || (n != LAST_N)) {
        // freeing memory if it is not "empty"
        // if (dev_A != NULL || dev_B != NULL || dev_C != NULL) {
            HANDLE_ERROR(cudaFree(dev_A));
            HANDLE_ERROR(cudaFree(dev_B));
            HANDLE_ERROR(cudaFree(dev_C));
        // }
        LAST_M = m;
        LAST_K = k;
        LAST_N = n;
        HANDLE_ERROR(cudaMalloc(&dev_A, size_A));
        HANDLE_ERROR(cudaMalloc(&dev_B, size_B));
        HANDLE_ERROR(cudaMalloc(&dev_C, size_C));
    }


    HANDLE_ERROR(cudaMemcpy(dev_A, A, size_A, cudaMemcpyHostToDevice)); 
    HANDLE_ERROR(cudaMemcpy(dev_B, B, size_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop; 
    float time = 0.0;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    gpu_blas_matmul(dev_A, dev_B, dev_C, m, k, n); 
    
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaEventRecord(stop, 0));
        
    HANDLE_ERROR(cudaMemcpy(C, dev_C, size_C, cudaMemcpyDeviceToHost));
    
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

    return time; // ms

}
