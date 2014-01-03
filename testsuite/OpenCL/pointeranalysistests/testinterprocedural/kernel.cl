//pass
//--local_size=1024 --num_groups=1024 --no-inline


void bar(__global float* pAlias)
{
  for(int i = 0; i < 100; i++) {
    pAlias[get_global_id(0)] = pAlias[get_global_id(0)] + 1.0f;
  }
}

void baz(__global float* qAlias)
{
  for(int i = 0; i < 100; i++) {
   qAlias[get_global_id(0)] = qAlias[get_global_id(0)] + 1.0f;
  }
}

__kernel void foo(__global float* p, __global float* q)
{

  bar(p);

  baz(q);

}
