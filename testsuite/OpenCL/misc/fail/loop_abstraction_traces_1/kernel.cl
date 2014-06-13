//xfail:BOOGIE_ERROR
//--local_size=16 --global_size=2048 --boogie-opt=/displayLoopAbstractions
//On entry to loop headed at .*kernel.cl:10:11:
//After 0 or more iterations of loop headed at .*kernel.cl:10:11

kernel void foo() {

    unsigned int i = 0;

    while(__invariant(i >= 0), __invariant(i <= 100), i < 100) {
        i++;
    }

    while(__invariant(i > 100), __invariant(i <= 200), i < 200) {
        i++;
    }

}
