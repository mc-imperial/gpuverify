//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=512
//Write by thread 200
//Write by thread 15
//x = 185

__global__ void example(float * A, int x) {
    __requires(x >= 0);
    __requires(x < 1000);

    if(threadIdx.x == 15) {
        A[threadIdx.x + x] = threadIdx.x;
    }

    if(threadIdx.x == 200) {
        A[threadIdx.x] = threadIdx.x;
    }
}
