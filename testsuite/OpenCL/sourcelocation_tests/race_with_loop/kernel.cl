//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=2 --no-inline
//Write by work item [\d]+ with local id [\d]+ in work group [\d], .+kernel.cl:16:[\d]+:
//Read by work item [\d]+ with local id [\d]+ in work group [\d], possible sources are:
//kernel.cl:12:(14|9)
//kernel.cl:13:(14|9)

__kernel void foo(__local int* p) {

    int x = 0;
    for(int i = 0; i < 100; i++) {
        x += p[i];
        x += p[i+1];
    }

    p[get_local_id(0)] = x;

}
