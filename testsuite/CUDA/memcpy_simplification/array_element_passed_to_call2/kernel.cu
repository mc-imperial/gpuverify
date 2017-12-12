//pass
//--blockDim=256 --gridDim=128

struct s {
  int a;
};

__device__ void bar(s x);

__global__ void foo(s *x)
{
  bar(x[1]);
}
