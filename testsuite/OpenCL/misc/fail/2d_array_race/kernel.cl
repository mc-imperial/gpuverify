//xfail:NOT_ALL_VERIFIED
//--local_size=[64,64] --global_size=[256,256]
//kernel.cl: error: possible write-read race on L\[\d+\]\[\d+\]
                                                         
kernel void example(global int *G) {

    local int L[64][64];

    L[get_local_id(1)][get_local_id(0)] = G[get_global_id(1)*get_global_size(1) + get_global_id(0)];

    L[get_local_id(1)+1][get_local_id(0)]++;

    G[get_global_id(1)*get_global_size(1) + get_global_id(0)] = L[get_local_id(1)][get_local_id(0)];

}
