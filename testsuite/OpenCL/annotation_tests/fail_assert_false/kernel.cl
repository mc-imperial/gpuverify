//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:7:3:[\s]+error:[\s]+this assertion might not hold for thread \([\d]+, [\d]+, [\d]+\) group \([\d]+, [\d]+, [\d]+\)[\s]+__assert\(false\);

__kernel void foo()
{
  __assert(false);
}
