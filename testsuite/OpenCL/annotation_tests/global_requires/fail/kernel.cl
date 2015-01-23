//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline



void baz(int x) {

}

void bar(int x) {
  __global_requires(x == 100);
}

__kernel void foo() {

  volatile int z = 50;
  if((get_local_id(0) % 2) == 0) {
    z = 100;
    bar(z);
  }

  baz(z);

}
