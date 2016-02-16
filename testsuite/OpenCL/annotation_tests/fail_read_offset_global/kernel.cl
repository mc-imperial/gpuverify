//xfail:NOT_ALL_VERIFIED
//--local_size=2048 --num_groups=4 --no-inline
//kernel.cl:12:[\d]+:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ with local id [\d]+ in work group [\d]+[\s]+__assert\(__read_offset_bytes\(p\) == 42\);






__kernel void foo(__global int* p) {

  __assert(__read_offset_bytes(p) == 42);

}
