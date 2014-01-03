//pass
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo(__local float* p, __local float* q)
{

  __local float* pAlias;

  __local float* qAlias;

  pAlias = p;

  qAlias = q;

  for(int i = 0; i < 100; i++) {
    pAlias[get_local_id(0)] = pAlias[get_local_id(0)] + 1.0f;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(int i = 0; i < 100; i++) {
   qAlias[get_local_id(0)] = qAlias[get_local_id(0)] + 1.0f;
  }

}
