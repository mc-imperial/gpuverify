//pass
//--blockDim=2048 --gridDim=64

struct s {
  float *p;
};

struct t : s {
};

__global__ void foo(t q) {
  __requires_fresh_array(q.p);
  q.p[threadIdx.x + blockIdx.x * blockDim.x] = 4.2f;
}
