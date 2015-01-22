//PASS
//--blockDim=1024 --gridDim=1 --no-inline

struct wrapped {
  unsigned int bidx;
  unsigned int bdim;
  unsigned int tidx;
};

__device__ float multiplyByTwo(float *v, wrapped tw)
{
    return 0.0f;
}

__device__ float divideByTwo(float *v, wrapped tw)
{
    return 0.0f;
}

typedef float(*funcType)(float*, wrapped);

__global__ void foo(float *v, funcType f, unsigned int size)
{
    __requires(f == multiplyByTwo | f == divideByTwo);
}
