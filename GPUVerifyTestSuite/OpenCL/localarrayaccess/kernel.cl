//pass
//--local_size=64 --num_groups=64


__kernel void foo() {

  __local int A[10];
  
  A[get_local_id(0)] = 2;

  barrier(CLK_LOCAL_MEM_FENCE);

  int x = A[get_local_id(0) + 1];

}