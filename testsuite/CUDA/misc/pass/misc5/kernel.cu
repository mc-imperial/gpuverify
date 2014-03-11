//pass
//--blockDim=2 --gridDim=1

__device__ int nondet(void);

__global__ void
crazy() 
{
    if (nondet()) {
        return;
    }
    while(nondet()) {
        while (nondet()) {
            while (nondet()) {
                if (nondet()) {
                    goto RECORD_RESULT;
                }
            }
        }
    RECORD_RESULT:
        (void)1;
    }
}
