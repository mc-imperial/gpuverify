//xfail:BOOGIE_ERROR
//--local_size=[64,64] --global_size=[256,256]
//kernel.cl: error: possible write-write race on L\[1\]\[2\]\[3\]\[3\]
                                                         
kernel void example(global int *G) {
    local int L[2][3][4][5];

    L[1][2][3][3] = G[get_global_id(0)];

}
