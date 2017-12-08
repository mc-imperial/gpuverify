//pass
//--local_size=1024 --num_groups=2

struct a {
  int i;
  short s;
};

__kernel void foo(__global struct a *q) {
  q[get_global_id(0)].i = 42;
  q[get_global_id(0)].s = 43;
}
