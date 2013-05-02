//pass
//--local_size=64 --num_groups=64


__kernel void foo(float a, float b) {

  int x;

  if (a != b) {
    x = 2;
  }
 
}


