//pass
//--local_size=64 --num_groups=64 --no-inline


__kernel void foo(int x) {

  if(x == 10) {
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
