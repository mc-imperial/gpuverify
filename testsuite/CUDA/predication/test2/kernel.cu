//pass
//--blockDim=2 --gridDim=1

__device__ char nondet();

__global__ void foo()
{
   while(nondet())
   {
      char c = nondet();
      while (c)
      {
        c = nondet();

        while (nondet())
        {
          if (nondet())
          {
            goto RECORD_RESULT;
          }
        }
      }

      RECORD_RESULT:

      ;
   }
}
