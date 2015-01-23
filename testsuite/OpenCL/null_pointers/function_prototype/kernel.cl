//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//error: possible null pointer access for work item

float* bar(float* p);

__kernel void foo()
{
  float x = 0;
  float *y = bar(&x);
  y[0] = y[0] + 1;
}
