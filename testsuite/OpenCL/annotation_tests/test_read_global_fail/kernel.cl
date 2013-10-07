//xfail:BOOGIE_ERROR
//--local_size=128 --num_groups=128
//kernel.cl:13:[\d]+:[\s]+error:[\s]+this assertion might not hold for thread [\d]+ in group [\d]+[\s]+__assert\(!__read\(q\)\);





__kernel void foo(__global int* p, __global int* q) {

  p[get_global_id(0)] = q[get_global_id(0)];

  __assert(!__read(q));

}
