//pass
//--local_size=64 --num_groups=64



__kernel void foo() {

  int4 a, b, c, d, e, f, g;

  a = (int4)(0, 1, 2, 3);

  __assert(a.x + a.y + a.z + a.w == 6);

  b = (int4)((int2)(0, 1), 2, 3);

  __assert(b.x + b.y + b.z + b.w == 6);

  c = (int4)((int2)(0, 1), (int2)(2, 3));

  __assert(c.x + c.y + c.z + c.w == 6);

  d = (int4)(0, (int2)(1, 2), 3);

  __assert(d.x + d.y + d.z + d.w == 6);

  e = (int4)(0, 1, (int2)(2, 3));

  __assert(e.x + e.y + e.z + e.w == 6);
  
  f = (int4)((int3)(0, 1, 2), 3);

  __assert(f.x + f.y + f.z + f.w == 6);

  g = (int4)(0, (int3)(1, 2, 3));

  __assert(g.x + g.y + g.z + g.w == 6);

  int i = a.x + b.y + c.z + d.w + e.x + f.y + g.z;

  __assert(i == 9);

}