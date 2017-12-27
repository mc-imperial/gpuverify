//pass
//--blockDim=2048 --gridDim=64

__global__ void foo(int *r) {
  r[threadIdx.x + blockIdx.x * blockDim.x] = warpSize;
}
