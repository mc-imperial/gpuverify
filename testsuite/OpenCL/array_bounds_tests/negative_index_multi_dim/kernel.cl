//pass
//--local_size=1,10 --num_groups=1,10 --check-array-bounds

__kernel void foo() {
  local int L[10][10];
  L[-1 * get_local_id(0)][get_global_id(1)] = get_global_size(0);
}
