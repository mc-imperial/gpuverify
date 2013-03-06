//pass
//--local_size=64 --num_groups=64


__kernel void foo(int j)
{
    int x = (bool)j ? 1 : 0;
}

