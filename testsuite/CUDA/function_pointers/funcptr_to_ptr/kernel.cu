//pass
//--blockDim=1024 --gridDim=1 --boogie-file=axioms.bpl --no-inline

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
    __requires(i != 0);
    __requires(f == divideByTwo);
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    void *x = (void*)f;
    if (i == 0)
      x = NULL;

    funcType g = (funcType)x;

    if (tid < size)
    {
        (*g)(v, tid);
    }
}
