//xfail:CLANG_ERROR
//--local_size=4 --num_groups=2
//kernel.cl:6:3:[\s]+error: implicit declaration of function 'foo' is invalid

__kernel void k() {
  foo();
}