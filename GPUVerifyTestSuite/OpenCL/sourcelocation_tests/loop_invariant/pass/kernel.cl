//pass
//--local_size=64 --num_groups=64


__kernel void foo() {

  int x = 0;

  while(__invariant(x <= 100), x < 100)
  {
    x = x + 1;
  }

}


