//xfail:BOOGIE_ERROR
//--blockDim=32 --gridDim=1 --warp-sync=32 --only-warp
__global__ void onlywarp_fail (int* A) {
	A[0] = threadIdx.x;
}
