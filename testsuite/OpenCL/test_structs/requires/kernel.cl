//pass
//--local_size=2048 --num_groups=64

struct s {
  int a;
};

__kernel void foo(struct s q) {
  __requires(q.a == 5);
  __assert(q.a == 5);
}
