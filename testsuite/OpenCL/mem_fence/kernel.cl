//pass
//--local_size=1024 --num_groups=24

__kernel void foo() {
  mem_fence(CLK_LOCAL_MEM_FENCE & CLK_GLOBAL_MEM_FENCE);
  read_mem_fence(CLK_LOCAL_MEM_FENCE & CLK_GLOBAL_MEM_FENCE);
  write_mem_fence(CLK_LOCAL_MEM_FENCE & CLK_GLOBAL_MEM_FENCE);
}
