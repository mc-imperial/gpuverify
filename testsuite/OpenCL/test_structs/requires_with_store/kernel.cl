//pass
//--local_size=2048 --num_groups=64

struct s {
  int a;
  int b;
};

__kernel void foo(__global int *p, struct s q) {
  __requires(q.a == 5);
  q.b = p[get_global_id(0)];
  __assert(q.a == 5);
}
