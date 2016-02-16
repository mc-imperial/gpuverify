//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --no-inline
//kernel.cl:7:3:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ with local id [\d]+ in work group [\d]+[\s]+__assert\(false\);

__kernel void foo()
{
  __assert(false);
}
