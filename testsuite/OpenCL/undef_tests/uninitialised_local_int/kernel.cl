//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:9:3:[\s]+error:[\s]+this assertion might not hold for thread [\d]+ in group [\d]+[\s]+__assert\(x == 0\);



__kernel void foo() {
  int x;
  __assert(x == 0);
}


