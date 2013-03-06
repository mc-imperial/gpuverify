//pass
//--local_size=1024 --num_groups=1024



__kernel void foo(__global int* p, __global int* q) {

  p[get_global_id(0)] = q[get_global_id(0)];

  __assert(!__write(q));

}