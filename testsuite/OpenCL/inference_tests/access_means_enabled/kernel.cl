//pass
//--local_size=32 --num_groups=32

kernel void foo() {

    local int A[32];

    A[get_local_id(0)] = 42;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j = 0; __global_invariant(__no_read(A)), __global_invariant(__no_write(A)), j < get_group_id(0); j++) {
      for(int k = 0; k < get_group_id(0); k++) {
        if(j > get_local_id(0)) {
            for(int i = 0; i < get_group_id(0); i++) {
                A[get_local_id(0)] += 42;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }

    A[(get_local_id(0) + 1) % 32] = 1;
}
