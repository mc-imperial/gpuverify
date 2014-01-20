//pass
//--local_size=1024 --num_groups=1 --no-infer --k-induction-depth=1

__kernel void foo(__local int *p) {

    for(int d = get_local_size(0) / 2;
        d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) < d) {
            p[get_local_id(0)] += p[get_local_id(0) + d];
        }
    }

    if(get_local_id(0) == 0) {
        p[0] = get_local_id(0);
    }
    

}
