//pass
//--local_size=64 --num_groups=128

kernel void foo(global float* p) {

    local float mine[64];

    // Src pointers are uniform within a group
    event_t handle = async_work_group_copy(mine, p + get_local_size(0)*get_group_id(0), get_local_size(0), 0);

    wait_group_events(1, &handle);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // Dst pointers are uniform within a group
    handle = async_work_group_copy(p + get_local_size(0)*get_group_id(0), mine, get_local_size(0), 0);

    wait_group_events(1, &handle);
    
}
