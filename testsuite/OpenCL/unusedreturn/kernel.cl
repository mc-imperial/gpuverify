//pass
//--local_size=64 --num_groups=64 --no-inline
int bar() {

  return 0;

}

__kernel void foo() {

  bar();

}
