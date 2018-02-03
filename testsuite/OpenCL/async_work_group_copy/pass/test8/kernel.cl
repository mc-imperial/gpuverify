//pass
//--local_size=64 --num_groups=128

#define N 64

kernel void foo(global float4* p, global float4* q) {

    local float4 mine1[N];
    local float4 mine2[N];

    global float4 *r;
    local float4 *mine;

    if (get_group_id(0) == 1) {
      r = p;
      mine = mine1;
    } else {
      r = q;
      mine = mine2;
    }

    event_t handle = async_work_group_copy(mine, r + N*get_group_id(0), N, 0);

    wait_group_events(1, &handle);
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    r[get_global_id(0)] = 2*mine[get_local_id(0)];

}
