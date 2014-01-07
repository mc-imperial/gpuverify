//pass
//--blockDim=1024 --gridDim=1 --no-inline

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foo(float *v, funcType f, unsigned int size)
{
    __requires(f == multiplyByTwo | f == divideByTwo);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        float x = (*f)(v, tid);
        x += multiplyByTwo(v, tid);
    }
}
