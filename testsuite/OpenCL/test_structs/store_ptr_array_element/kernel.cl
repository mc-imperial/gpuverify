//pass
//--local_size=2048 --num_groups=64

struct s {
  __global int *p[5];
};

__kernel void foo(__global int *p, struct s q) {
  q.p[3] = p;
  q.p[3][get_global_id(0)] = 5;
}
