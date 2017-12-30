//pass
//--blockDim=2048 --gridDim=64

struct s {
  float *p[42];
};

__global__ void foo(s q) {
  __requires_fresh_array(q.p[4]);
  q.p[4][threadIdx.x + blockIdx.x * blockDim.x] = 4.2f;
}
