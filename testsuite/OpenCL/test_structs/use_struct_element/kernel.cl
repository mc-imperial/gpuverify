//pass
//--local_size=2048 --num_groups=64

struct s {
  int _;
  struct t {
   int a;
  } a;
};

__kernel void foo(__global int *p, struct s q) {
  p[get_global_id(0)] = q.a.a;
}
