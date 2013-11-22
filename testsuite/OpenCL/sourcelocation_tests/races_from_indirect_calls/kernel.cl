//xfail:BOOGIE_ERROR
//--local_size=16 --num_groups=2
//kernel.cl: error: possible read-write race on \(\(char\*\)p\)
//Write by thread [\d]+ in group \d, .+kernel.cl:39:7
//p\[tid \+ 1\] = tid;
//Read by thread [\d]+ in group \d, possible sources are:
//kernel.cl:18:11
//kernel.cl:19:11
//kernel.cl:20:11
//kernel.cl:29:14
//kernel.cl:29:27
//kernel.cl:34:14

#define tid get_local_id(0)

int jazz(__local int *x, __local int *y, __local int *z) {
    return
        x[tid] +
        y[tid + 1] +
        z[tid + 1];
}
          
int sim(int x, __local int * b) {
    jazz(b, b, b);
    return 0;
}
        
int bar(__local int* a) {
    return a[tid] + sim(a[tid + 2], a);
}

int foo(__local int* p) {
    bar(p);
    return p[tid];
}

__kernel void baz(__local int* p) {
    int x = foo(p);
    p[tid + 1] = x + tid;
}
