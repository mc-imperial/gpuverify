//pass
//--local_size=64 --num_groups=64


__axiom(get_local_size(0)==64);
__axiom(get_num_groups(0)==1);

__kernel void foo(__local int* A, int i) {
  A[i] = get_local_id(0) / 64;
}