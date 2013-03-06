//pass
//--local_size=64 --num_groups=64



__kernel void foo() {

  int2 v;

  v.y = 1;

  int i;

  i = v.y;

  __assert(i == 1);

}