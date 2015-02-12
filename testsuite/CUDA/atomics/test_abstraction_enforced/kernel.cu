//xfail:NOT_ALL_VERIFIED
//--blockDim=2 --gridDim=1
//assert

__global__ void f(int *c) {

    if(threadIdx.x == 0) {
        *c = 0;
        atomicAdd(c, 1);
        int x = *c;
        int y = *c;
        __assert(x == 0);
        __assert(y == 0);
    }
}
