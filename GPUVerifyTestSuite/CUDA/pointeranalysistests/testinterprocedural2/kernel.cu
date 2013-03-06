//pass
//--blockDim=64 --gridDim=64
#include "cuda.h"

__device__ void bar (int *p){
  
    int a;

    p = &a;
}


__global__ void foo (int* p, int* q){

    if (*p > 10){
        bar(p);
    }
    else {
        bar(q);
    }
}
