//xfail:BOOGIE_ERROR
//--local_size=[64] --global_size=[256]
//kernel.cl: error: possible write-write race on L\[1\]\[2\]\[3\]\[3\] \(byte 2\)
                                                         
kernel void example(global int *G) {
    local int L[2][3][4][5];

    if(get_global_id(0) == 0) {
        L[1][2][3][3] = G[get_global_id(0)];
    }

    local char* p = (local char*)&(L[0][0][0][0]);

    if(get_global_id(0) == 1) {
        p[sizeof(int)*(60+2*20+3*5+3) + 2] = 42;
    }
    

}
