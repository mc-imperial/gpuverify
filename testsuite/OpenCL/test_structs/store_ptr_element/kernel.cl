//pass
//--local_size=2048 --num_groups=64

struct s {
  __global int *p;
};

__kernel void foo(__global int *p, struct s q) {
  q.p = p;
  q.p[get_global_id(0)] = 5;
}
