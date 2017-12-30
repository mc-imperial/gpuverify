//xfail:NOT_ALL_VERIFIED
//--blockDim=2048 --gridDim=64
//possible write-write race on q.p\[0\]

struct s {
  char *p;
};

__global__ void foo(s q) {
  __requires_fresh_array(q.p);
  q.p[0] = threadIdx.x;
}
