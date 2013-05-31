//pass
//--local_size=1024 --num_groups=1024

__kernel void example(__local int * A) {

    for(unsigned i = 0; i < 100; i++) {
        for(unsigned j = 0; j < (1 << 16); j += 16) {
            __assert((j % 16) == 0);
        }
    }

}
