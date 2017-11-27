//pass
//--local_size=32 --num_groups=32

__kernel void foo(int x, int y, int z)
{
    for (int i = 0; i < (x ? y : z); i++) {
        __assert(i >= 0);
    }
}

