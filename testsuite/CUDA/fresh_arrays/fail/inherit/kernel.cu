//xfail:NOT_ALL_VERIFIED
//--blockDim=2048 --gridDim=64
//possible write-write race on q.p\[0\]

struct s {
  float *p;
};

struct t : s {
};

__global__ void foo(t q) {
  __requires_fresh_array(q.p);
  q.p[0] = threadIdx.x;
}
