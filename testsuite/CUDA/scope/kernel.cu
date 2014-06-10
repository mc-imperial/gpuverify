//pass
//--blockDim=64 --gridDim=64 --no-inline
#include <cuda.h>

#define BIN_COUNT 64
#define THREAD_N 64

__global__ void k(unsigned int *d_Result) {

    //Per-thread histogram storage
    __shared__ unsigned int s_Hist[THREAD_N * BIN_COUNT];

    //Flush shared memory
    for(int i = 0; 
        __invariant(__uniform_bool(__enabled())),
        __invariant(__uniform_int(i)),
        __invariant(i >= 0),
        __invariant(__implies(__write(s_Hist), (((__write_offset_bytes(s_Hist)/sizeof(unsigned int)) % THREAD_N) - threadIdx.x) == 0)),
        i < BIN_COUNT / 4; i++) {
      s_Hist[threadIdx.x + i * THREAD_N] = 0;
    }

    __syncthreads();

    if(threadIdx.x < BIN_COUNT){
        unsigned int sum = 0;
        const int value = threadIdx.x;

        const int valueBase = (value * THREAD_N);
        const int  startPos = (threadIdx.x & 15) * 4;

        for(int i = 0, accumPos = startPos; i < THREAD_N; i++){
            sum += s_Hist[valueBase + accumPos];
            accumPos++;
            if(accumPos == THREAD_N) accumPos = 0;
        }

        d_Result[blockIdx.x * BIN_COUNT + value] = sum;
    }

}
