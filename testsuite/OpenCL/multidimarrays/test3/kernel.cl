//pass
//--local_size=64 --num_groups=64 --no-inline
__kernel void foo() {

  __local int A[100][99][98];

  int x = A[5][4][3]; // We should generate A + 99*(100*5 + 4) + 3

}
