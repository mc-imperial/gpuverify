//pass
//--local_size=64 --num_groups=64


__kernel void foo(int x)
{
    while (x + 100 < 102) {
      if(get_local_id(0) < 5) {
        x = 2;
        break;
      }
    }
}