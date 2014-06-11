//xfail:BOOGIE_ERROR
//--local_size=64 --global_size=256
//kernel.cl: error: possible write-write race on G\[3\] \(bytes 4..7\)
                                                         
kernel void example(global float4 *G) {

    G[3].y = (float)get_global_id(0);

}
