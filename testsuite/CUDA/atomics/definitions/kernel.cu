//pass
//--blockDim=1024 --gridDim=1

#include <cuda.h>

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
  atomicAdd(A,10);
  atomicSub(A,10);
  atomicExch(A,10);
  atomicMin(A,10);
  atomicMax(A,10);
  atomicAnd(A,10);
  atomicOr(A,10);
  atomicXor(A,10);
  atomicCAS(A,10,11);

  atomicAdd(B,10);
  atomicSub(B,10);
  atomicExch(B,10);
  atomicMin(B,10);
  atomicMax(B,10);
  atomicAnd(B,10);
  atomicOr(B,10);
  atomicXor(B,10);
  atomicInc(B,10);
  atomicDec(B,10);
  atomicCAS(B,10,11);

  atomicAdd(C,10);
  atomicExch(C,10);
  atomicMin(C,10);
  atomicMax(C,10);
  atomicAnd(C,10);
  atomicOr(C,10);
  atomicXor(C,10);
  atomicCAS(C,10,11);

  atomicAdd(D,10.0);
  atomicExch(D,10.0);
}
