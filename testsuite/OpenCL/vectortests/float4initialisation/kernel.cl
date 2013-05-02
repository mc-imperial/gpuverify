//pass
//--local_size=64 --num_groups=64


static __attribute__((always_inline)) bool __equal_floats(float* p, float* q) {
  char* cp = (char*)p;
  char* cq = (char*)q;
  return cp[0] == cq[0] &&
         cp[1] == cq[1] &&
         cp[2] == cq[2] &&
         cp[3] == cq[3];
}

__kernel void foo() {

  float4 a, b, c, d, e, f, g;

  a = (float4)(0, 1, 2, 3);

  b = (float4)((float2)(4, 5), 6, 7);

  c = (float4)((float2)(8, 9), (float2)(10, 11));

  d = (float4)(12, (float2)(13, 14), 15);

  e = (float4)(16, 17, (float2)(18, 19));
  
  f = (float4)((float3)(20, 21, 22), 23); 

  g = (float4)(24, (float3)(25, 26, 27));

  float _0 = 0;
  float _1 = 1;
  float _2 = 2;
  float _3 = 3;
  float _4 = 4;
  float _5 = 5;
  float _6 = 6;
  float _7 = 7;
  float _8 = 8;
  float _9 = 9;
  float _10 = 10;
  float _11 = 11;
  float _12 = 12;
  float _13 = 13;
  float _14 = 14;
  float _15 = 15;
  float _16 = 16;
  float _17 = 17;
  float _18 = 18;
  float _19 = 19;
  float _20 = 20;
  float _21 = 21;
  float _22 = 22;
  float _23 = 23;
  float _24 = 24;
  float _25 = 25;
  float _26 = 26;
  float _27 = 27;

  __assert(__equal_floats((float*)&a + 0, &_0));
  __assert(__equal_floats((float*)&a + 1, &_1));
  __assert(__equal_floats((float*)&a + 2, &_2));
  __assert(__equal_floats((float*)&a + 3, &_3));

  __assert(__equal_floats((float*)&b + 0, &_4));
  __assert(__equal_floats((float*)&b + 1, &_5));
  __assert(__equal_floats((float*)&b + 2, &_6));
  __assert(__equal_floats((float*)&b + 3, &_7));

  __assert(__equal_floats((float*)&c + 0, &_8));
  __assert(__equal_floats((float*)&c + 1, &_9));
  __assert(__equal_floats((float*)&c + 2, &_10));
  __assert(__equal_floats((float*)&c + 3, &_11));

  __assert(__equal_floats((float*)&d + 0, &_12));
  __assert(__equal_floats((float*)&d + 1, &_13));
  __assert(__equal_floats((float*)&d + 2, &_14));
  __assert(__equal_floats((float*)&d + 3, &_15));

  __assert(__equal_floats((float*)&e + 0, &_16));
  __assert(__equal_floats((float*)&e + 1, &_17));
  __assert(__equal_floats((float*)&e + 2, &_18));
  __assert(__equal_floats((float*)&e + 3, &_19));

  __assert(__equal_floats((float*)&f + 0, &_20));
  __assert(__equal_floats((float*)&f + 1, &_21));
  __assert(__equal_floats((float*)&f + 2, &_22));
  __assert(__equal_floats((float*)&f + 3, &_23));

  __assert(__equal_floats((float*)&g + 0, &_24));
  __assert(__equal_floats((float*)&g + 1, &_25));
  __assert(__equal_floats((float*)&g + 2, &_26));
  __assert(__equal_floats((float*)&g + 3, &_27));

  float i = a.x + b.y + c.z + d.w + e.x + f.y + g.w;

}