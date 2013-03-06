//pass
//--local_size=64 --num_groups=64


__kernel void foo() {
  float4 a;
  float4 b;
  float4 c;
  float4 r = select(a, b, c <= 0.1f);
}
