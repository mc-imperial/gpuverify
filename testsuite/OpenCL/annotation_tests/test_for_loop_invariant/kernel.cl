//pass
//--local_size=[64,64] --num_groups=[64,64,64] --no-inline


__kernel void foo() {

  for(int d = 1; 
    __invariant(__uniform_bool(__enabled())),
    __invariant(__uniform_int(d)),
    __invariant(__implies(__enabled(), d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64)),
      d < 64; d <<= 1) {
  }

}
