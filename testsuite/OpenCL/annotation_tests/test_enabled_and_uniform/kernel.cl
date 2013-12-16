//pass
//--local_size=64 --num_groups=64


__kernel void foo(__global int* p)
{
  for(int k = 0;
          __assert(__uniform_int(k)),
          __assert(__uniform_bool(__enabled())),
      k < 1000; k++)
  {
    ;
    ;
  }
}
