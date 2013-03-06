//pass
//--local_size=64 --num_groups=64

__kernel void foo() {

  int x[10];

  x[4] = 10;

  barrier(CLK_LOCAL_MEM_FENCE);

  int temp = x[4];

  __assert(temp == 10);

}