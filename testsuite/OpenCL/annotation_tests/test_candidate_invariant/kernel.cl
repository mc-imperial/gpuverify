//pass
//--local_size=[64,64] --num_groups=[64,64,64]


__kernel void foo() {

  for(int d = 1; 
    __candidate_invariant(false),
    __candidate_invariant(d == 3),
    __candidate_invariant(__uniform_bool(__enabled())),
    __candidate_invariant(__uniform_int(d)),
    __candidate_invariant(__implies(__enabled(), d == 1 | d == 2 | d == 4 | d == 8 | d == 16 | d == 32 | d == 64)),
      d < 64; d <<= 1) {
  }

}