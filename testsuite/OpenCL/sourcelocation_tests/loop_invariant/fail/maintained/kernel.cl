//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64
//kernel.cl:10:9:[\s]+error:[\s]+loop invariant might not be maintained by the loop for work item [\d]+ in work group [\d]+[\s]+while\(__invariant\(x < 100\), x < 100\)


__kernel void foo() {

  int x = 0;

  while(__invariant(x < 100), x < 100)
  {
    x = x + 1;
  }

}
