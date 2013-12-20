//pass
//--blockDim=64 --gridDim=64 --no-inline
#include "cuda.h"

__device__ void bar (int *p){
  
    int a;

    p = &a;
}


__global__ void foo (int* p, int* q){

    bar(p);

    bar(q);
}
