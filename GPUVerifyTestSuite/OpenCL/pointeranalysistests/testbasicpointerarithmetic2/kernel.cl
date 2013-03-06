//pass
//--local_size=8192 --num_groups=64


__kernel void foo(__local float* p, __local float* q)
{

  __local float* pAlias;

  pAlias = p + 10;

  for(int i = 0; i < 100; i++) {
    pAlias[get_local_id(0)] = pAlias[get_local_id(0)] + 1.0f;
  }

}
