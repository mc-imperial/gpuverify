//pass
//--local_size=64 --num_groups=64 --no-inline



__kernel void foo() {

  float4 v;

  v.x = 4.0f;

  float f;

  f = v.x;

}
