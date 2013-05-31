//pass
//--local_size=1024 --num_groups=1024

__kernel void example(__local int * A) {

    for(unsigned i = 0; i < (1 << 16); i += 128) {
        __assert((i % 128) == 0);
        for(unsigned j = 0; j < (1 << 16); j += 16) {
            __assert((j % 16) == 0);
            for(unsigned k = 3; k < (1 << 16); k += 32) {
                __assert((k % 32) == 3);
            }
        }
    }

}
