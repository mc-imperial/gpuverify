//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(int j)
{
    int x = (bool)j ? 1 : 0;
}

