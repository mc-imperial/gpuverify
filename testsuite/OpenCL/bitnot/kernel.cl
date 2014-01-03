//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo() {
  int x, y;
  x = 2;
  y = ~x;
}
