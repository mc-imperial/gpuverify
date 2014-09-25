//pass
//--blockDim=2 --gridDim=2

struct S {
  int * p;
};

__global__ void foo(int * A) {

  S myS;
  myS.p = A;
  int * q;
  q = myS.p;
  q[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;

}
