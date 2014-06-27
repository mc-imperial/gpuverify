//pass
//--local_size=64 --num_groups=64 --no-inline

__constant int3 c[2] = {{0, 0, 0}, {1, 0, 0}};

__kernel void foo()
{
  __assert(c[0].x == 0);
  __assert(c[1].x == 1);
}
