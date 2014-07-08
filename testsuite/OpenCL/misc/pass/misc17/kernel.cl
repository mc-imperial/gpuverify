//PASS
//--local_size=2 --num_groups=2

#define __max(X, Y) ((X) > (Y) ? (X) : Y)

kernel void foo(int x) {

    for(int i = __max(0, -32*x); i < 100; i++) {
        __assert(i >= __max(0, -32*x));
    }

}
