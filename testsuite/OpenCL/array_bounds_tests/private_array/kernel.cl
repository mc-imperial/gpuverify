//pass
//--local_size=4 --num_groups=4 --check-array-bounds

__kernel void foo() {
  int L[16];
  int x = get_global_id(0);
  L[x] = x * x;
}
