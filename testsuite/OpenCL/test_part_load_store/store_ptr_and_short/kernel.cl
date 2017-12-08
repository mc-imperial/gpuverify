//pass
//--local_size=1024 --num_groups=2

struct a {
  __global int *p;
  short i;
};

__kernel void foo(__global int *p, __global struct a *q) {
  q[get_global_id(0)].i = 42;
  q[get_global_id(0)].p = p;
}
