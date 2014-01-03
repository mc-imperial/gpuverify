//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo() {

  unsigned x = 0;

  while(__invariant(x <= 4), x < 4)
  {
    x = x + 1;
  }

  __assert(x == 4);

}


