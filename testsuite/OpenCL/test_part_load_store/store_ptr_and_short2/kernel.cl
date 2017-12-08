//pass
//--local_size=1024 --num_groups=2

struct a {
  __global int *p;
  short2 i;
};

__kernel void foo(__global int *p, __global struct a *q) {
  q[get_global_id(0)].i = (short2){42, 43};
  q[get_global_id(0)].p = p;
}
