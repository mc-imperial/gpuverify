//xfail:NOT_ALL_VERIFIED
//--local_size=16 --num_groups=2 --no-inline
//possible write-read race
//Read by work item.+kernel.cl:16:(9|11)
//Write by work item.+from external source location


void __spec_foo(__local int * p) {
    __writes_to(p);
}

void baz(int x);

__kernel void bar(__local int * A) {
    __spec_foo(A);
    baz(A[get_local_id(0)]);
}

