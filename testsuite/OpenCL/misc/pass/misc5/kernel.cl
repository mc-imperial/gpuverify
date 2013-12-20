//pass
//--num_groups=1024 --local_size=1024 --no-inline

__kernel void foo()
{
    volatile int4 a = (int4)(1, 2, 3, 0);
    volatile int4 b = (int4)(2, 3, 4, 1);
    int4 c = min(a, b);
    __assert(c.x == 1);
    __assert(c.y == 2);
    __assert(c.z == 3);
    __assert(c.w == 0);
}
