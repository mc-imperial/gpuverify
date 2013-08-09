//xfail:BOOGIE_ERROR
//--local_size=2048 --num_groups=4
//kernel.cl:12:[\d]+:[\s]+error:[\s]+this assertion might not hold for thread \([\d]+, [\d]+, [\d]+\) group \([\d]+, [\d]+, [\d]+\)[\s]+__assert\(__read_offset\(p\) == __read_offset\(q\)\);






__kernel void foo(__global int* p, __global int* q) {

  __assert(__read_offset(p) == __read_offset(q));

}
