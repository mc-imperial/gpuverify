//pass
//--local_size=1024 --num_groups=64 --no-inline


__attribute__((always_inline)) inline void bar(__local float* pAlias)
{
  for(int i = 0;
    __invariant(__implies(__read(pAlias), __read_offset(pAlias)/sizeof(float) == get_local_id(0))),
    __invariant(__implies(__write(pAlias), __write_offset(pAlias)/sizeof(float) == get_local_id(0))),
     i < 100; i++) {
    pAlias[get_local_id(0)] = pAlias[get_local_id(0)] + 1.0f;
  }
}

__attribute__ ((always_inline)) inline void baz(__local float* qAlias)
{
  for(int i = 0;
    __invariant(__implies(__read(qAlias), __read_offset(qAlias)/sizeof(float) == get_local_id(0))),
    __invariant(__implies(__write(qAlias), __write_offset(qAlias)/sizeof(float) == get_local_id(0))),
    i < 100; i++) {
   qAlias[get_local_id(0)] = qAlias[get_local_id(0)] + 1.0f;
  }
}

__kernel void foo(__local float* p, __local float* q)
{

  bar(p);

  barrier(CLK_LOCAL_MEM_FENCE);

  baz(q);

}
