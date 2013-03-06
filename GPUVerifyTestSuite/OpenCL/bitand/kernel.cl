//pass
//--local_size=64 --num_groups=64


__kernel void foo() {
  int x, y, z;
  x = 0;
  y = 1;
  z = x & y;
}