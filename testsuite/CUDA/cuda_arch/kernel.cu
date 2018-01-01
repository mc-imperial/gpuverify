//pass
//--blockDim=2048 --gridDim=64

#if __CUDA_ARCH__ < 350
#error Unexpected __CUDA_ARCH__
#endif

__global__ void foo() {
}
