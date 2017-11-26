//pass
//--local_size=2048 --num_groups=64

struct s {
  __global int *p;
};

__kernel void foo(__global int *p, struct s q) {
  __requires(q.p == p);
  p[get_global_id(0)] = q.p[get_global_id(0)];
}
