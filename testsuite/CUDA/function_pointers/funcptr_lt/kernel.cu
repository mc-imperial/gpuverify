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

__global__ void foog(float *v, funcType f, funcType g, unsigned int size)
{
    __requires(f == divideByTwo);
    __requires(g == multiplyByTwo);
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    funcType h;
    if (f >= g)
      h = f;
    else
      h = g;

    if (tid < size)
    {
        (*h)(v, tid);
    }
}
