//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:9:3:[\s]+error:[\s]+this assertion might not hold for thread [\d]+ in group [\d]+[\s]+__assert\(false\);


__kernel void foo()
{
  __assume(true);
  __assert(false);
}