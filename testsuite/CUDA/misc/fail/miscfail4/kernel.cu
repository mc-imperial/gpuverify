//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1
//kernel.cu: error: possible read-write race
//GPUVerify kernel analyser finished with 0 verified, 1 error

// In CUDA providing static and __attribute__((always_inline)) should not
// keep a copy of inlined function around.

#define tid threadIdx.x

__device__ static __attribute__((always_inline))
void inlined(int *A, int offset)
{
   int temp = A[tid + offset];
   A[tid] += temp;
}

__global__ void inline_test(int *A, int offset) {
  inlined(A, offset);
}
