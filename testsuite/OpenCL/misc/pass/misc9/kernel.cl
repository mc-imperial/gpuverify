//pass
//--local_size=16 --num_groups=16 --no-inline

void baz(int);

__kernel void foo(__local int* p, __local int* q, int x) {
    __local int * r;

    r = x ? p : q;

    baz(__read_offset_bytes(r));
    
}
