//pass
//--local_size=1024 --num_groups=2 --no-inline

__kernel void atomicTest(__local long *A, int B)
{
   A[get_local_id(0) + 1] = 42;
   atomic_add((__local int*)A, B);
}
