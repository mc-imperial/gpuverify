//pass
//--blockDim=2 --gridDim=1

__device__ char nondet();

__global__ void foo()
{
  while (nondet())
  {
    char r; // r not initialised

    if (r != nondet())
    {
        goto RECORD_RESULT;
    }
  }

  RECORD_RESULT:

  ;
}
