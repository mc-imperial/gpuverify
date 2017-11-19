//pass
//--blockDim=1024 --gridDim=1 --no-inline

typedef int(*funcType)(int);
typedef char(*funcTypeC)(int);

__device__ int multiplyByTwo(int i)
{
    __ensures(__return_val_int() == i * 2);
    return i * 2;
}

__device__ int divideByTwo(int i)
{
    return i / 2;
}

__global__ void foo(int *v, char *w)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    funcType f = (tid != 0) ? multiplyByTwo : divideByTwo;

    v[tid] = (*f)(tid);
    __assert(__implies(tid != 0, v[tid] == 2 * tid));
}

