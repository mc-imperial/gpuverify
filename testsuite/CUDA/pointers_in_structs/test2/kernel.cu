//pass
//--blockDim=2 --gridDim=2

struct S {
  int * p;
};

__global__ void foo(struct S A) {
  A.p[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;
}
