//pass
//--local_size=64 --num_groups=64 --no-inline


int4 bar(int4 v) {
  __ensures(__return_val_int4().x == v.x + 1);
  __ensures(__return_val_int4().y == v.y + 1);
  __ensures(__return_val_int4().z == v.z + 1);
  __ensures(__return_val_int4().w == v.w + 1);
  int4 result;
  result.x = v.x + 1;
  result.y = v.y + 1;
  result.z = v.z + 1;
  result.w = v.w + 1;
  return result;
}


__kernel void foo() {

  int4 a;

  a = (int4)(0, 1, 2, 3);

  int4 b = bar(a);

  __assert(b.x == 1);
  __assert(b.y == 2);
  __assert(b.z == 3);
  __assert(b.w == 4);

}
