//xfail:NOT_ALL_VERIFIED
//--local_size=16 --num_groups=2 --no-inline
//kernel.cl: error: possible read-write race on p
//Write by work item [\d]+ with local id [\d]+ in work group \d, .+kernel.cl:39:[\d]+
//p\[tid \+ 1\] = tid;
//Read by work item [\d]+ with local id [\d]+ in work group \d, possible sources are:
//kernel.cl:18:(9|11)
//kernel.cl:19:(9|11)
//kernel.cl:20:(9|11)
//kernel.cl:29:(12|14)
//kernel.cl:29:(25|27)
//kernel.cl:34:(12|14)

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
    foo(p);
    p[tid + 1] = tid;
}
