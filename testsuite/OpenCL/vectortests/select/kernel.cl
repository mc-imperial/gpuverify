//pass
//--local_size=64 --num_groups=64

float4 bar(void);

__kernel void foo() {
  float4 a = bar();
  float4 b = bar();
  float4 c = bar();
  float4 r = select(a, b, c <= 0.1f);
}
