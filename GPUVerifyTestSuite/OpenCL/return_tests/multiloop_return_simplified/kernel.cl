//pass
//--local_size=64 --num_groups=64


__kernel void bar(int y) {

  int result = 0;

  for(int j = 0; j < 100; j++)
  {
    if(result > 1000) {
      return;
    }
    result += j;
  }

  for(int j = 0; j < 100; j++)
  {
    result += j;
  }

}