//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1

kernel void foo(global int* __restrict p, global int* __restrict q) {

    local int mydata[1024];

    event_t handle = async_work_group_copy(mydata, p, 1024, 0);

    // This barrier is not enough to protect the subsequent
    // read from the async_work_group_copy
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    q[get_global_id(0)] = mydata[get_global_id(0)];

    wait_group_events(1, &handle);

}
