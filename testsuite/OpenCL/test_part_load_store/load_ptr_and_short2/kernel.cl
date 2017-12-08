//pass
//--local_size=1024 --num_groups=2

struct a {
  __global int *p;
  short2 i;
};

__kernel void foo(__global struct a *q1,
                  __global struct a *q2) {
  q2[get_global_id(0)].i = q1[get_global_id(0)].i;
  q2[get_global_id(0)].p = q1[get_global_id(0)].p;
}
