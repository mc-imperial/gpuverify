//pass
//--local_size=64 --num_groups=64



__kernel void foo() {

  int4 a, b, c, d, e, f, g;

  a = (int4)(0, 1, 2, 3);

  __assert(a.x + a.y + a.z + a.w == 6);

  b = (int4)((int2)(4, 5), 6, 7);

  __assert(b.x + b.y + b.z + b.w == 22);

  c.x = 10;
  c.y = c.x;
  c.z = c.x;
  c.w = 2*c.x;

  __assert(c.x + c.y + c.z + c.w == 50);

  d = (int4)(0, (int2)(1, 2), 3);

  d.x = d.w;

  __assert(d.x + d.y + d.z + d.w == 9);

  e = d;

  e.w++;

  __assert(e.x + e.y + e.z + e.w == 10);
  
  f = (int4)((int3)(e.w, e.z, e.y), 3);

  __assert(f.x + f.y + f.z + f.w == 10);

  g = (int4)(0, (int3)(1, 2, 3));

  g.w = g.x;

  g.z = g.x;

  g.y = g.x;

  __assert(g.x + g.y + g.z + g.w == 0);

}