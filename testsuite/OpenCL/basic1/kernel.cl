//pass
//--local_size=64 --num_groups=64

#include "opencl.h"

__kernel void foo() {
  int a, b, c;
  a = 2;
  b = 3;
  c = a + b;
}