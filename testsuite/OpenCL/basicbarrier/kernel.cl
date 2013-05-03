//pass
//--local_size=64 --num_groups=64


__kernel void foo() {
  int a, b, c;
  a = 2;
  b = 3;
  c = a + b;
  barrier(CLK_LOCAL_MEM_FENCE);
}