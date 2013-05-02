//pass
//--local_size=1024 --num_groups=1024



void bar(int x) {
  __global_requires(x == 100);
}

__kernel void foo() {
  int z = 100;

  if((get_local_id(0) % 2) == 0) {
    bar(z);
  }

}