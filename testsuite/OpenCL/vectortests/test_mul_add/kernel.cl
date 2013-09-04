//xfail:BOOGIE_ERROR
//--local_size=64 --num_groups=64

//Note: this test looks like it should pass.  However, it fails because
//clang generates a muladd intrinsic which breaks floating point associativity.
//We keep the test (and mark it as xfail) because it is good to know that
//GPUVerify can pass through the intrinsic to Boogie

__attribute__((always_inline)) inline bool __equal_floats(float* p, float* q) {
  char* cp = (char*)p;
  char* cq = (char*)q;
  return cp[0] == cq[0] &&
         cp[1] == cq[1] &&
         cp[2] == cq[2] &&
         cp[3] == cq[3];
}

__kernel void foo() {

  float4 r;
  float4 acc;
  float s;

  r = (float4)(1, 2, 3, 4);
  acc = (float4)(1, 1, 1, 1);

  s = 2;

  acc += r*s;

  float t1 = 1;
  float t2 = 2;
  float t3 = 3;
  float t4 = 4;

  float res1 = t1 + t1*t2;
  float res2 = t1 + t2*t2;
  float res3 = t1 + t3*t2;
  float res4 = t1 + t4*t2;

  __assert(__equal_floats((float*)&acc + 0, &res1));
  __assert(__equal_floats((float*)&acc + 1, &res2));
  __assert(__equal_floats((float*)&acc + 2, &res3));
  __assert(__equal_floats((float*)&acc + 3, &res4));

}
