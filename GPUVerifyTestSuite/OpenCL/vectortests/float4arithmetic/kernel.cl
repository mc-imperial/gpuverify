//pass
//--local_size=64 --num_groups=64


__axiom (get_local_size(0) == 4096);
__axiom (get_num_groups(0) == 1024);

static __attribute__((always_inline)) bool __equal_floats(float* p, float* q) {
  char* cp = (char*)p;
  char* cq = (char*)q;
  return cp[0] == cq[0] &&
         cp[1] == cq[1] &&
         cp[2] == cq[2] &&
         cp[3] == cq[3];
}

__kernel void foo(__global float4* p) {

  float4 v = p[get_global_id(0)];

  float4 w = v + v;

  float v2x = v.x + v.x;
  float v2y = v.y + v.y;
  float v2z = v.z + v.z;
  float v2w = v.w + v.w;

  __assert (__equal_floats((float*)(&w) + 0, &v2x));
  __assert (__equal_floats((float*)(&w) + 1, &v2y));
  __assert (__equal_floats((float*)(&w) + 2, &v2z));
  __assert (__equal_floats((float*)(&w) + 3, &v2w));

  p[get_global_id(0)] = w;

}
