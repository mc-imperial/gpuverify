//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//error: possible null pointer access for work item

float* bar(float* p)
{
  p[0] += 1;
  return p;
}

__kernel void foo(int i)
{
  float x = 0;
  float *y;

  if (i == 0)
    y = bar(0);
  else
    y = bar(&x);

  x += 1;
}
