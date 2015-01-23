//xfail:NOT_ALL_VERIFIED
//--blockDim=512 --gridDim=1 --no-infer --no-inline
//kernel.cu:57:21: error: loop invariant might not be maintained

#include <cuda.h>

__global__ void helloCUDA(
    int num32PathPerBlock,
    int callPutFlag,
    float faceValue,
    float strike,
    float dt,
    int actualTimeSteps,
    int numTimeOffsets,
    float *volMat,
    float *driftMat,
    float *f0)
{
    __requires(numTimeOffsets == 143);
    __requires(actualTimeSteps == 13);

    __shared__ float sdata[256*32];
    __shared__ float sdata2[32];
    __shared__ float s_f0[256];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int numWarpsPerBlock = blockDim.x / 32;

    __shared__ float N1[32];

    __shared__ float discount[32];
    __shared__ float optionPrice[32];
    __shared__ float zeroRate[32*16];
    __shared__ float zeroBondPrice[32];

    __shared__ float s_volMat[256];
    __shared__ float s_driftMat[256];

    float localZeroRate = 0;
    float localSigma;
    float drift;
    volatile float f_j;

    int numPointsPerCurve;

    unsigned int sidx, local_i, local_j;

    /*--------one MC run---------*/
        __syncthreads();

        numPointsPerCurve = numTimeOffsets;

            for(int j = 0;
                    //__invariant(0 <= j), //< the missing invariant
                    __invariant(__implies(__write(sdata), ((__write_offset_bytes(sdata)/sizeof(float)) % blockDim.x) == threadIdx.x)),
                    __invariant(!__read(sdata)),
                    __invariant(__implies(__write(sdata), (__write_offset_bytes(sdata)/sizeof(float)) < (j/16)*blockDim.x)),
                    __invariant((j % 16) == 0),
                    j < numPointsPerCurve;
                    j += 16)
            {
                __assert(j < 143);
                f_j = sdata[(j + (threadIdx.x / 32) + 1) * 32 + (threadIdx.x % 32)];
                __syncthreads();
                sdata[      (j + (threadIdx.x / 32)    ) * 32 + (threadIdx.x % 32)] = f_j;
            }

}
