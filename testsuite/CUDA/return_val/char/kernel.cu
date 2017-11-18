//xfail:BUGLE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: Type of __return_val function does not match return type

__device__ char multiplyByTwo(int i)
{
    __ensures(__return_val_int() == i * 2);
    return i * 2;
}
