//xfail:NOT_ALL_VERIFIED
//--local_size=16 --num_groups=2 --no-inline
//possible read-write race
//Write by work item [\d]+ with local id [\d]+ in work group \d, .+kernel.cl:21:[\d]+
//p\[tid \+ 1\] = tid
//Read by work item [\d]+ with local id [\d]+ in work group \d, .+kernel.cl:12:(12|14)
//return A\[tid\]

#define tid get_local_id(0)

int foo(__local int * A) {
    return A[tid];
}

__kernel void bar(__local int * p) {

    for(int i = 0; i < 100; i++) {
        foo(p);
    }

    p[tid + 1] = tid;

}

