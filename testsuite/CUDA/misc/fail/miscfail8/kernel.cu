//xfail:NOT_ALL_VERIFIED
//--blockDim=2 --gridDim=2
//assert

typedef struct {
  unsigned int a, b;
} pair;

__device__ void assertion(pair A) {
  __assert(false);
}

__global__ void test(pair A)
{
  assertion(A);
}
