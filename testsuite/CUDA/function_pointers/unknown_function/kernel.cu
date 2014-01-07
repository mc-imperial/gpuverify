//xfail:BUGLE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: Unsupported function pointer

typedef double(*funcType)(double);

__device__ double bar(double x) {
  return sin(x);
}

__global__ void foo(double x, int i)
{
  funcType f;

  if (i == 0)
    f = bar;
  else
    f = cos;

  f(x);
}
