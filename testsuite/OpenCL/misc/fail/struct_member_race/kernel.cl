//xfail:BOOGIE_ERROR
//--local_size=64 --global_size=256
//kernel.cl: error: possible write-write race on G\[3\] \(bytes 4..7\)

typedef struct {
    int x;
    int y;
} S;

kernel void example(global S *G) {

    G[3].y = get_global_id(0);

}
