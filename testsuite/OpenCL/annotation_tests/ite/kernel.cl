//pass
//--local_size=1024 --num_groups=1024 --no-inline



__kernel void foo(int x)
{
  __local int S[1024*1024];

  int index = (x ? get_local_id(0) : get_global_id(0));

  for(int i = 0;
       __invariant(__no_read(S)),
       __invariant(__write_implies(S, __write_offset(S)/sizeof(int) == __ite(x, get_local_id(0), get_global_id(0)))),
       __invariant(__ite(x, __write_implies(S, __write_offset(S)/sizeof(int) == get_local_id(0)), __write_implies(S, __write_offset(S)/sizeof(int) == get_global_id(0)))),
       i < 100; i++) {
       S[index] = get_global_id(0);
  }


}
