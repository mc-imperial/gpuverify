//pass
//--local_size=1024 --num_groups=1024


__kernel void foo(__global float* p, __global float* q)
{

  __global float* pAlias;

  __global float* qAlias;

  pAlias = p;

  qAlias = q;

  for(int i = 0; i < 100; i++) {
    pAlias[get_global_id(0)] = pAlias[get_global_id(0)] + 1.0f;
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  for(int i = 0; i < 100; i++) {
   qAlias[get_global_id(0)] = qAlias[get_global_id(0)] + 1.0f;
  }

}
