//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//possible attempt to modify constant memory

__constant__ int A[1024];

__global__ void foo(int *B) {
  A[threadIdx.x] = B[threadIdx.x];
}
