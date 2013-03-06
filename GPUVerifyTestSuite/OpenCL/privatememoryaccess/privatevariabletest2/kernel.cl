//pass
//--local_size=64 --num_groups=64

__kernel void foo() {

  int x;

  x = 10;

  int* p;

  p = &x;

  *p = 20;

  __assert(x == 20);

}