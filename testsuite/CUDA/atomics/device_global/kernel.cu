//pass
//--gridDim=1 --blockDim=2 --only-divergence

__device__ unsigned int x = 0;

__global__ void f()
{
    atomicInc(&x, 1);
}
