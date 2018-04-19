//pass
//--local_size=1024 --num_groups=2 --no-inline

__kernel void atomicTest(__local short *A, int B)
{
   A[get_local_id(0) + 2] = 42;
   atomic_add((__local int*)A, B);
}
