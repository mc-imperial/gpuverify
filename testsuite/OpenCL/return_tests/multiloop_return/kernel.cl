//pass
//--local_size=64 --num_groups=64 --no-inline


int bar(int y) {

  int result = 0;

  for(int j = 0; j < 100; j++)
  {
    result += j;
  }

  for(int k = 0; k < 100; k++)
  {
    result += k;
    for(int w = 0; w < get_local_id(0); w++) {
      result += w;
      if(result > 1000)
      {
         return 0;
      }
    }
  }

  for(int q = 0; q < 100; q++)
  {
    result += q;
  }

  return result;

}

__kernel void foo()
{
  int x;
  for(int i = 0; i < 200; i++)
  {
    x = bar(i);
  }
  return;
}
