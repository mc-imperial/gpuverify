//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:10:9:[\s]+error:[\s]+loop invariant might not hold on entry for thread [\d]+ in group [\d]+[\s]+while\(__invariant\(x <= 100\), x < 100\)


__kernel void foo() {

  int x = 101;

  while(__invariant(x <= 100), x < 100)
  {
    x = x + 1;
  }

}