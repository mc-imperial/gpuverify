//pass
//--local_size=64 --num_groups=64
__kernel void foo() {

  __local int A[10][5];

  int x = A[3][4]; // We should generate A + 10*3 + 4

}