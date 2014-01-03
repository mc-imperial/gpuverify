//pass
//--local_size=1024 --num_groups=1024 --no-inline


void bar(__global float* a, __global float* b, __global float* c)
{
  a[get_global_id(0) + 1] = get_global_id(0);
  b[get_global_id(0) + 1] = get_global_id(0);
  c[get_global_id(0) + 1] = get_global_id(0);
}

void baz(__global float* a, __global float* b, __global float* c)
{
  a[get_global_id(0) + 1] = get_global_id(0);
  b[get_global_id(0) + 1] = get_global_id(0);
  c[get_global_id(0) + 1] = get_global_id(0);
}

__kernel void foo(__global float* p, __global float* q, __global float* r, __global float* s, __global float* t, __global float* u, int x)
{
  __global float* temp;

  if(x > 4) {
    temp = p;
  } else {
    temp = q;
  }

  temp[get_global_id(0)] = get_global_id(0);

  bar(r, s, t);
  bar(t, u, s);

  baz(r, r, t);
  baz(t, s, t);

}
