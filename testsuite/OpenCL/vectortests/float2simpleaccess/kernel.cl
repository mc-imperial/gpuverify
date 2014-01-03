//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void k() {
  float2 v;
  float s;
  v.x = 42.0f;
  v.y = 42.0f;
  s = v.x;
  s = v.y;
}
