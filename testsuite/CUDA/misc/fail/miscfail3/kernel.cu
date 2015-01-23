//xfail:NOT_ALL_VERIFIED
//--blockDim=1024 --gridDim=1 --no-inline
//kernel.cu: error: possible read-write race
//GPUVerify kernel analyser finished with 1 verified, 1 error

// In CUDA providing the inline keyword should still keep a copy of
// the function around (contrary to OpenCL). However, by default a
// function with this keyword is not actually inlined at the optimisation
// level used by GPUVerify.

#define tid threadIdx.x

__device__ inline void inlined(int *A, int offset)
{
   int temp = A[tid + offset];
   A[tid] += temp;
}

__global__ void inline_test(int *A, int offset) {
  inlined(A, offset);
}
