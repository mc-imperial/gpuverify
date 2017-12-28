//pass
//--blockDim=2048 --gridDim=64

struct s {
  float x, y, z;
};

__global__ void foo(s *q) {
  s p = { 0.0f, 0.0f, 0.0f };
  q[threadIdx.x + blockIdx.x * blockDim.x] = p;
}
