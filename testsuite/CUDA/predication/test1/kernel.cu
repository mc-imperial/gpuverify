//pass
//--blockDim=2 --gridDim=1

__device__ char nondet();

__global__ void foo() {
  while(nondet())
  {
    while (nondet())
    {

    }

  }
}
