//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=4
//error: this assertion might not hold for thread

__constant__ int global_constant[4];

__global__ void constant(int *in) {
    global_constant[threadIdx.x] = in[threadIdx.x];
}
