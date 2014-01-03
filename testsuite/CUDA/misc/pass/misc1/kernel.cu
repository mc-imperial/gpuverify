//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void helloCUDA(int x)
{
    __requires(x == 143);
    __shared__ float S[256*32];
    __shared__ float F[256];

    unsigned int idx;

    //initialise data on shared memory
    for(int i = 0;
            __invariant(__implies(__write(S), ((__write_offset(S)/sizeof(float)) % blockDim.x) == threadIdx.x)),
            i < x;
            i += (blockDim.x/32))
    {
        if((i+(threadIdx.x/32)) < x){
            idx = (i+(threadIdx.x/32))*32+(threadIdx.x%32);
            S[idx] = F[i+(threadIdx.x/32)];
        }
    }
}
