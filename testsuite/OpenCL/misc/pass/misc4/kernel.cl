//pass
//--num_groups=1024 --local_size=1024

__kernel void foo(float3 x, float3 y)
{
  float3 temp = fmin(x,y);
}
