//pass
//--blockDim=1024 --gridDim=1 --no-inline

#define tid (blockIdx.x * blockDim.x + threadIdx.x)

__device__ void multiplyByTwo(float *v, unsigned int index)
{
    __requires(__ptr_offset(v) == 0);
    __requires(index == tid);
    __requires(__read_implies(v, __read_offset(v)/sizeof(float) == index));
    __requires(__write_implies(v, __write_offset(v)/sizeof(float) == index));
    __ensures(__read_implies(v, __read_offset(v)/sizeof(float) == index));
    __ensures(__write_implies(v, __write_offset(v)/sizeof(float) == index));
    v[index] = v[index] * 2.0f;
}

__device__ void divideByTwo(float *v, unsigned int index)
{
    __requires(__ptr_offset(v) == 0);
    __requires(index == tid);
    __requires(__read_implies(v, __read_offset(v)/sizeof(float) == index));
    __requires(__write_implies(v, __write_offset(v)/sizeof(float) == index));
    __ensures(__read_implies(v, __read_offset(v)/sizeof(float) == index));
    __ensures(__write_implies(v, __write_offset(v)/sizeof(float) == index));
    v[index] = v[index] * 0.5f;
}

typedef void(*funcType)(float*, unsigned int);

__global__ void foo(float *v, unsigned int i)
{
    __requires(i == 1 | i == 2);
    funcType f;

    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    (*f)(v, tid);
}
