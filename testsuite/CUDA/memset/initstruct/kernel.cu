//pass
//--gridDim=1 --blockDim=32

__global__ void kernel(uint4 *out) {
  uint4 vector = {0,0,0,0};
  out[threadIdx.x] = vector;
}
