//pass
//--local_size=4 --num_groups=64



__kernel void foo() {

  unsigned x = 0;

  while(__global_invariant(__implies(!__enabled(), x == get_local_id(0))),
        __invariant(x <= get_local_id(0)), x < get_local_id(0))
  {
    x = x + 1;
  }

  __assert(x == get_local_id(0));

}


