//pass
//--local_size=[32,32] --num_groups=8 --no-inline





__kernel void foo() {

  __local int A[32][32];
  
  A[get_local_id(1)][get_local_id(0)] = 2;

}
