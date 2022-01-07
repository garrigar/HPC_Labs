#include <cuda.h>
#include <iostream>
#include "vectorsum.h"

#define GPU_ID 1

#define THREADS_PER_BLOCK 256

// int *dev_in = NULL, *dev_out = NULL; 
// int *host_out = NULL;
// int LAST_IN_BYTES_COUNT = -1;

static void checkError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess)
    {
        fprintf(stderr, "Error: %s:%d, ", file, line);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
        exit( EXIT_FAILURE );
    }
}

#define CHECK_ERROR( err ) ( checkError( err, __FILE__, __LINE__ ) )


__device__ void warpReduce(volatile int *sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduceSum(int *g_idata, int *g_odata)
{

    // using shared memory
    extern __shared__ int sdata[];

    // Performing first level of reduction while reading from global memory
    // and writing to shared memory at the same time.
    // The number of blocks "needed" can now be halved.
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // no need to sync here due to the warp size of 32
    if (tid < 32)
        warpReduce(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

timed_result gpu_vectorsum(int *arr, int n) {
    cudaSetDevice(GPU_ID);

    int blocks_count_twice = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // Halving number of blocks "needed", as was discussed.
    int blocks_count = (blocks_count_twice + 1) / 2;

    size_t in_bytes_count = n * sizeof(int);
    size_t out_bytes_count = blocks_count * sizeof(int);

    int *dev_in = NULL, *dev_out = NULL; 
    int *host_out = NULL;

    CHECK_ERROR(cudaMalloc(&dev_in, in_bytes_count));
    CHECK_ERROR(cudaMalloc(&dev_out, out_bytes_count));
    host_out = new int[blocks_count];

    CHECK_ERROR(cudaMemcpy(dev_in, arr, in_bytes_count, cudaMemcpyHostToDevice));
    
    size_t shared_mem_bytes_count = THREADS_PER_BLOCK * sizeof(int);   

    cudaEvent_t start, stop; 
    float time = 0.0;

    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));
    CHECK_ERROR(cudaEventRecord(start, 0));
    
    reduceSum<<< blocks_count, THREADS_PER_BLOCK, shared_mem_bytes_count >>>(dev_in, dev_out);
    
    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());
    CHECK_ERROR(cudaEventRecord(stop, 0));

    CHECK_ERROR(cudaMemcpy(host_out, dev_out, out_bytes_count, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaEventElapsedTime(&time, start, stop)); // records in 'time' the time in ms

    // final summing
    int out = 0;
    for (size_t i = 0; i < blocks_count; i++)
    {
        out += host_out[i];
    }

    timed_result ans;
    ans.result = out;
    ans.time = time;

    // cleanup
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));
    CHECK_ERROR(cudaFree(dev_in));
    CHECK_ERROR(cudaFree(dev_out));
    delete[] host_out;

    return ans;    
}
