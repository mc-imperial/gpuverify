//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64 --no-inline
//kernel.cl:9:3:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ in work group [\d]+[\s]+__assert\(false\);


__kernel void foo()
{
  __assume(true);
  __assert(false);
}
