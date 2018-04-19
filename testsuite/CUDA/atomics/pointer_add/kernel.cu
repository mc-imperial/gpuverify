//pass
//--gridDim=2 --blockDim=1024

char *buffer;

__global__ void atomicTest(int B)
{
   atomicAdd((unsigned int *)&buffer, B);
}

