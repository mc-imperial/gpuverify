//pass
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float* p) {

    local float mine[2*N];
    local float mine_again[2*N];
    
    event_t handle = async_work_group_copy(mine, p, N, 0);

    global char* q = (global char*)p;

    async_work_group_copy((local char*)mine, q, sizeof(float)*N, handle);

    mine[N + get_local_id(0)] = get_global_id(0);
    mine_again[N + get_local_id(0)] = get_global_id(0);
    
    wait_group_events(1, &handle);

    p[get_global_id(0)] = 2*mine[get_local_id(0)];

}
