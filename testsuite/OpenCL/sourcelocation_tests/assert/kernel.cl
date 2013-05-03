//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:15:3:[\s]+error:[\s]+this assertion might not hold for thread \([\d]+, [\d]+, [\d]+\) group \([\d]+, [\d]+, [\d]+\)[\s]+__assert\(x == 101\);


__kernel void foo() {

  int x = 0;

  while(__invariant(x <= 100), x < 100)
  {
    x = x + 1;
  }
  
  __assert(x == 101);

}