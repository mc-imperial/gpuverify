//PASS
//--blockDim=1024 --gridDim=1 --no-inline

struct wrapped {
  unsigned int bidx;
  unsigned int bdim;
  unsigned int tidx;
};

__device__ float multiplyByTwo(float *v, wrapped tw)
{
    unsigned int tid = tw.bidx * tw.bdim + tw.tidx;
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, wrapped tw)
{
    unsigned int tid = tw.bidx * tw.bdim + tw.tidx;
    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, wrapped);

__global__ void foo(float *v, funcType f, unsigned int size)
{
    __requires(f == multiplyByTwo | f == divideByTwo);

    wrapped tid = {blockIdx.x,  blockDim.x, threadIdx.x};

    if ((tid.bidx * tid.bdim + tid.tidx) < size)
    {
        float x = (*f)(v, tid);
    }
}
