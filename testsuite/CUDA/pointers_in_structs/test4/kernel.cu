//pass
//--blockDim=2 --gridDim=2

struct S {
  struct {
    int * p;
    int * q;
  } s;
};

__global__ void foo(struct S A) {
  A.s.p[threadIdx.x + blockDim.x*blockIdx.x] = 
   A.s.q[threadIdx.x + blockDim.x*blockIdx.x] + threadIdx.x;
  A.s.q[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;
}
