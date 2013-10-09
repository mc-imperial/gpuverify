//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//error: possible null pointer access for thread

__kernel void foo(int i)
{
  float x = 0;
  float *y;

  if (i)
   y = &x;
  else
   y = NULL;


  y[0] = y[0] + 1;
}

