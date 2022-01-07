#ifndef _VECTORSUM_H_
#define _VECTORSUM_H_

typedef struct
{
    int result;
    float time;
} timed_result;

timed_result gpu_vectorsum(int *arr, int n);

#endif
