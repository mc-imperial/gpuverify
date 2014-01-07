//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --boogie-file=axioms.bpl --no-inline
//error: possible null pointer access

typedef float(*funcType)(float*, unsigned int);

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

__global__ void foo(float *v, funcType f, unsigned int size, int i)
{
    __requires(f == divideByTwo);
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    int *x = (int*)f;
    if (i == 0)
      x += 5;

    funcType g = (funcType)x;

    if (tid < size)
    {
        (*g)(v, tid);
    }
}
