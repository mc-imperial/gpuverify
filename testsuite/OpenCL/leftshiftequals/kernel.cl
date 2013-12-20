//pass
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo() {
  unsigned int x = 1;
  signed int y = 2;
  
  x <<= x;
  y <<= x;
  
}
