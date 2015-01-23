//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//error: possible null pointer access for work item

__kernel void foo(int i)
{
  float x = 0;
  float *y;

  if (i)
   y = &x;
  else
   y = 0;


  y[0] = y[0] + 1;
}

