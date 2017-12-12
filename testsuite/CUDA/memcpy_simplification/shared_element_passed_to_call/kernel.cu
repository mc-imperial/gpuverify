//pass
//--blockDim=256 --gridDim=128

struct s {
  int a;
};

__device__ void bar(s x);

__global__ void foo()
{
  __shared__ s y[4];
  bar(y[3]);
}
