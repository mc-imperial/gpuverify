//pass
//--blockDim=256 --gridDim=128

texture<unsigned, 1, cudaReadModeElementType> texDKey128;

__global__ void foo()
{
    tex1Dfetch(texDKey128, 4);
}
