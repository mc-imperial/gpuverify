//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: possible null pointer access

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foor(float *v, unsigned int size, unsigned int i)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f;

    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    if (tid < size)
    {
        float x = (*f)(v, tid);
        x += multiplyByTwo(v, tid);
    }
}
