//pass
//--local_size=2048 --num_groups=4 --no-inline




__kernel void foo(__global int* p, __global int* q) {

  p[get_global_id(0)] = q[get_global_id(0)];

  __assert(!__read(p));

}
