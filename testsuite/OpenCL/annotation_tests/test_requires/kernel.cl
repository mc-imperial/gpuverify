//pass
//--local_size=64 --num_groups=[64,64] --no-inline

void bar(int x) {
  __requires(x > 0);

}

__kernel void foo() {

  int d = 1;

  while(
    __invariant(__uniform_bool(__enabled())),
        __invariant(__uniform_int(d)),
    __invariant(__implies(__enabled(), d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64)),
      d < 64) {
    bar(d);
    d <<= 1;
  }

}
