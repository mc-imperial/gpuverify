//xfail:BOOGIE_ERROR
//--local_size=2048 --num_groups=4
//kernel.cl:12:[\d]+:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ in work group [\d]+[\s]+__assert\(__read_offset\(p\) == 42\);






__kernel void foo(__global int* p) {

  __assert(__read_offset(p) == 42);

}
