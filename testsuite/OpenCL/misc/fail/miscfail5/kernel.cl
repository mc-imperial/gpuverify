//xfail:BOOGIE_ERROR
//--num_groups=1024 --local_size=1024
//assert\(c.x == 2\)
__kernel void foo()
{
    volatile int4 a = (int4)(1, 2, 3, 0);
    volatile int4 b = (int4)(2, 3, 4, 1);
    int4 c = min(a, b);
    __assert(c.x == 2);
}
