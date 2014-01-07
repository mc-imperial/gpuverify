//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: possible null pointer access

#define tid (blockIdx.x * blockDim.x + threadIdx.x)

__device__ float multiplyByTwo(float *v, unsigned int index)
{
    return v[index] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int index)
{
    return v[index] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foo(float *v)
{
    funcType f = (funcType)5;
    (*f)(v, tid);
}
