//xfail:NOT_ALL_VERIFIED
//--local_size=1,10 --num_groups=1,10 --check-array-bounds
//kernel.cl:7:[\d]+:[\s]+error:[\s]+possible array out-of-bounds access on array L

__kernel void foo() {
  local int L[10][10];
  L[-1 + get_global_id(0)][get_global_id(1)] = get_global_size(0);
}
