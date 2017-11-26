//pass
//--local_size=2048 --num_groups=64

struct s {
  struct t {
    int a;
  } a;
};

__kernel void foo(__global int *p, struct s q) {
  q.a.a = p[get_global_id(0)];
}
