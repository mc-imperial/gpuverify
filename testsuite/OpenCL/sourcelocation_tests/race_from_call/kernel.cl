//xfail:NOT_ALL_VERIFIED
//--local_size=16 --num_groups=2 --no-inline
//kernel.cl: error: possible read-write race on p
//Write by work item [\d]+ with local id [\d]+ in work group \d, .+kernel.cl:18:[\d]+
//p\[tid \+ 1\] = tid;
//Read by work item [\d]+ with local id [\d]+ in work group \d, .+kernel.cl:13:(12|14)
//return p\[tid\];


#define tid get_local_id(0)

int foo(__local int* p) {
    return p[tid];
}

__kernel void baz(__local int* p) {
    foo(p);
    p[tid + 1] = tid;
}


