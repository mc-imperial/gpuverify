//xfail:NOT_ALL_VERIFIED
//--local_size=16 --global_size=2048 --cruncher-opt=/displayLoopAbstractions
//On entry to loop headed at .*kernel.cl:15:11:
//After 0 or more iterations of loop headed at .*kernel.cl:15:11
//On entry to loop headed at .*kernel.cl:19:11:
//After 0 or more iterations of loop headed at .*kernel.cl:19:11
//On entry to loop headed at .*kernel.cl:23:11:
//After 0 or more iterations of loop headed at .*kernel.cl:23:11
//On taking back edge to head of loop at .*kernel.cl:23:11:

kernel void foo() {

    unsigned int i = 0;

    while(__invariant(i >= 0), __invariant(i <= 100), i < 100) {
        i++;
    }

    while(__invariant(i >= 100), __invariant(i <= 200), i < 200) {
        i++;
    }

    while(__invariant(i >= 200), __invariant(i < 300), i < 300) {
        i++;
    }
    
}
