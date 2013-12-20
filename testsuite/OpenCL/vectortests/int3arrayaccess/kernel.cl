//pass
//--local_size=2048 --num_groups=64 --no-inline



__kernel void foo(__global int* p) {

  __local int3 vs[10];

  vs[get_local_id(0)].x = p[get_global_id(0)];

  volatile int i;

  i = vs[get_local_id(0)].x;

}
