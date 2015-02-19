//pass
//--local_size=2 --num_groups=1 --kernel-args=foo,0x0000000300000003

struct pack {
  int a;
  int b;
};

kernel void foo (struct pack x) {
  __assert(x.a == 3);
  __assert(x.b == 3);
}
