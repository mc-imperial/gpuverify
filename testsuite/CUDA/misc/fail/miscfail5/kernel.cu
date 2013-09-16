//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=4
//attempt to modify constant memory

__constant__ int global_constant[4];

__global__ void constant(int *in) {
    global_constant[threadIdx.x] = in[threadIdx.x];
}
