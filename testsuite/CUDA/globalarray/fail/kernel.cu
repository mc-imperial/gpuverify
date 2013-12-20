//xfail:BOOGIE_ERROR
//--blockDim=8 --gridDim=1 --no-inline

// The statically given values for A are not preserved when we translate CUDA
// since the host is free to change the contents of A.
// cf. testsuite/OpenCL/globalarray/pass2

__constant__ int A[8] = {0,1,2,3,4,5,6,7};

__global__ void globalarray(float* p) {
  int i = threadIdx.x;
  int a = A[i];

  if(a != threadIdx.x) {
    p[0] = threadIdx.x;
  }
}
