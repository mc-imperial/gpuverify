//pass
//--blockDim=2 --gridDim=2

struct S {
  struct {
    int * p;
  } s;
};

__global__ void foo(struct S A) {
  A.s.p[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;
}
