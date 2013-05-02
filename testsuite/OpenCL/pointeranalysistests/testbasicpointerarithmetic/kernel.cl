//pass
//--local_size=256 --num_groups=128


__kernel void foo(__global float* p, __global float* q)
{

  __global float* pAlias;

  pAlias = p + 10;

  for(int i = 0; i < 100; i++) {
    pAlias[get_global_id(0)] = pAlias[get_global_id(0)] + 1.0f;
  }

}
