//xfail:NOT_ALL_VERIFIED
//--blockDim=1024 --gridDim=1
//null pointer access

__global__ void foo(int *H) {
  size_t tmp = (size_t)H;
  tmp += sizeof(int);
  int *G = (int *)tmp;
  G -= 1;
  G[threadIdx.x] = threadIdx.x;
}
