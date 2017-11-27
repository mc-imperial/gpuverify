//xfail:GPUVERIFYVCGEN_ERROR
//--local_size=256 --num_groups=2 --vcgen-opt=/checkArrays:AB
//Array name 'AB' specified for restricted checking is not found

kernel void foo(global int * A) {

}
