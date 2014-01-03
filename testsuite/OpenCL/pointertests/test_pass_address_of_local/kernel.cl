//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64 --no-inline
//kernel.cl:19:3:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ in work group [\d]+[\s]+__assert\(x == 4\);


void bar(__private int* x)
{
  *x = 5;
}

__kernel void foo()
{
  int x;

  x = 4;

  bar(&x);

  __assert(x == 4);

}

