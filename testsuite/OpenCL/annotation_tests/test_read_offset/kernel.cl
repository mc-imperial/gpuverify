//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo()
{
    __local float A[1024];
    __assert(__implies(__read(A), __read_offset_bytes(A) == 0));
}
