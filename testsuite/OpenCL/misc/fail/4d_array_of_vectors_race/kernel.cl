//xfail:BOOGIE_ERROR
//--local_size=[64,64] --global_size=[256,256]
//kernel.cl: error: possible write-write race on L\[1\]\[2\]\[3\]\[3\] \(bytes 8..11\)
                                                         
kernel void example(global float4 *G) {
    local float4 L[2][3][4][5];

    L[1][2][3][3].z = G[get_global_id(0)].x;

}
