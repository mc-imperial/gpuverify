//pass
//--blockDim=1024 --gridDim=1 --no-inline

typedef float(*funcType)(float*, unsigned int);

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

__device__ funcType grabFunction(int i) {
  __requires(i != 0);
  __ensures(__return_val_funptr(funcType) == divideByTwo);
  if (i == 0)
    return multiplyByTwo;
  else
    return divideByTwo;
}

__global__ void foo(float *v, unsigned int size, int i)
{
    __requires(i != 0);
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f = grabFunction(i);

    if (tid < size)
    {
        (*f)(v, tid);
    }
}
