//pass
//--blockDim=2048 --gridDim=2 --no-inline

__constant__ int A[4096];
__constant__ int B[3] = {0,1,2};

__global__ void kernel() {
  int x = A[threadIdx.x] + B[0];
}
