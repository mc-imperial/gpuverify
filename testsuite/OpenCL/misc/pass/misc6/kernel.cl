//pass
//--no-infer --num_groups=64 --local_size=16

#define TYPE float8

__kernel void foo(volatile __global TYPE *p)
{
    for(unsigned i = 0;
        __invariant(__write_implies(p, ((__write_offset(p)/sizeof(TYPE))%(get_global_size(0))) == get_global_id(0))),
        i < 100; i++) {
        p[get_global_size(0)*i + get_global_id(0)] = get_global_id(0);
    }
}
