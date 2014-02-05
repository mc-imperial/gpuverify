//pass
//--blockDim=32 --gridDim=32 --warp-sync=32 --only-warp
__global__ void onlywarp_pass (int* A) {
	A[threadIdx.x] = threadIdx.x;
	A[threadIdx.x+1] = threadIdx.x;
}
