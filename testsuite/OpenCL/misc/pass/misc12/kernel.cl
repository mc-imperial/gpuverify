//pass
//--no-infer --local_size=16 --num_groups=16

kernel void foo() {

    for(int i = 0; i < get_group_id(0); i++) {
        barrier(CLK_LOCAL_MEM_FENCE);
    }

}
