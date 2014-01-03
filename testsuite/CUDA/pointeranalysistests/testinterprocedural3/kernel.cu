//pass
//--blockDim=64 --gridDim=64 --no-inline
#include "cuda.h"

__device__ void baz (int p []){
    int a;

    p = &a;
}

__device__ void bar (int *p){
  
    int a;

    p = &a;
}


__global__ void foo (int* p, int* q){

    __shared__ int sharedArr  [100];

    __shared__ int sharedArr2 [50];

    bar(p);

    baz (sharedArr);

    bar(q);

    if (*q){
        baz(sharedArr2);
    }
}
