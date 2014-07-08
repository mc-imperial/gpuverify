//pass
//--local_size=32 --num_groups=32

kernel void foo() {

    for(int j = 0; j < get_group_id(0); j++) {
        if(j > get_local_id(0)) {
            for(int i = 0; i < get_group_id(0); i++) {
                __global_assert(__implies((j < get_group_id(0)) & (j > get_local_id(0)) & (i < get_group_id(0)), __enabled()));
            }
        }
    }
}
