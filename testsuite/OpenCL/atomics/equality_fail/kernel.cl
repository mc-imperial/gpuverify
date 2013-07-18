//xfail:GPUVERIFYVCGEN_ERROR
//--local_size=1024 --num_groups=1 --equality-abstraction
//GPUVerify: error: --equality-abstraction cannot be used with atomics\.

kernel void simple (local int* A)
{
  atomic_inc(A);
}
