//xfail:BOOGIE_ERROR
//--local_size=16 --num_groups=2
//kernel.cl: error: possible read-write race on \(\(char\*\)p\)
//Write by thread [\d]+ in group \d, .+kernel.cl:18:7
//p\[tid \+ 1\] = tid;
//Read by thread [\d]+ in group \d, .+kernel.cl:13:14
//return p\[tid\];


#define tid get_local_id(0)

int foo(__local int* p) {
    return p[tid];
}

__kernel void baz(__local int* p) {
    foo(p);
    p[tid + 1] = tid;
}


