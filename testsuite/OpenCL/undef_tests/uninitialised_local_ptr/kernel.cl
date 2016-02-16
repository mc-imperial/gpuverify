//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --clang-opt=-Wno-uninitialized --no-inline
//kernel.cl:9:3:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ with local id [\d]+ in work group [\d]+[\s]+__assert\(x == 0\);



__kernel void foo() {
  int* x;
  __assert(x == 0);
}


