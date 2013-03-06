//pass
//--local_size=1024 --num_groups=1024



__kernel void foo()
{

  for(int i = get_global_id(0); 
    __invariant(__implies(__uniform_bool(__enabled()), __distinct_int(i))),
    i < 100; i++)
  {
    ;

  }

}