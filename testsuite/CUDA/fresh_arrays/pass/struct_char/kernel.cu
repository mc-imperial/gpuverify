//pass
//--blockDim=2048 --gridDim=64

struct s {
  char *p;
};

__global__ void foo(s q) {
  __requires_fresh_array(q.p);
  q.p[threadIdx.x + blockIdx.x * blockDim.x] = 42;
}
