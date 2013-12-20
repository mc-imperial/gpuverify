//pass
//--local_size=2048 --num_groups=1111 --equality-abstraction --no-inline

#pragma OPENCL EXTENSION cl_khr_fp64: enable

__attribute((always_inline)) inline bool __equal_doubles(double* p, double* q) {
  char* cp = (char*)p;
  char* cq = (char*)q;

  return cp[0] == cq[0] &&
         cp[1] == cq[1] &&
         cp[2] == cq[2] &&
         cp[3] == cq[3] &&
         cp[4] == cq[4] &&
         cp[5] == cq[5] &&
         cp[6] == cq[6] &&
         cp[7] == cq[7];

}

__kernel void foo(__global double4* p) {

  double4 v = p[get_global_id(0)];

  double4 w = v + v;

  double v2x = v.x + v.x;
  double v2y = v.y + v.y;
  double v2z = v.z + v.z;
  double v2w = v.w + v.w;

  __assert(__equal_doubles(((double*)&w) + 0, &v2x));
  __assert(__equal_doubles(((double*)&w) + 1, &v2y));
  __assert(__equal_doubles(((double*)&w) + 2, &v2z));
  __assert(__equal_doubles(((double*)&w) + 3, &v2w));

  p[get_global_id(0)] = w;

}
