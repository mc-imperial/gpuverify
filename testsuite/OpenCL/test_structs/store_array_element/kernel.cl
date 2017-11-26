//pass
//--local_size=2048 --num_groups=64

struct s {
  int a[5];
};

__kernel void foo(__global int *p, struct s q) {
  q.a[3] = p[get_global_id(0)];
}
