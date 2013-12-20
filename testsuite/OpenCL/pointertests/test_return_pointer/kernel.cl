//pass
//--local_size=1024 --num_groups=64 --no-inline



__local int* bar(__local int* p)
{
  return p;
}

__kernel void foo(__local int* A)
{
  __local int* q;

  q = bar(A);

  q[get_local_id(0)] = get_local_id(0);

}

