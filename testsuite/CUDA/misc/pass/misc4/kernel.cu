//pass
//--gridDim=1024 --blockDim=1024 --no-inline

__device__ void bar(int i);

__global__ void
foo(int w, int h)
{
   __requires(h == 0);

   for (int i = threadIdx.x;
        __invariant(h == 0),
        i < w; i += blockDim.x)
   {
     if (h == 0)
       bar(5);
     else
       __assert(0);
   }
}
