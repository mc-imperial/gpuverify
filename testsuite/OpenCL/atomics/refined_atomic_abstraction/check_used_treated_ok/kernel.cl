//pass
//--local_size=64 --num_groups=12

__kernel void foo(__global unsigned * c, __global char * g, size_t n) {
    volatile int x = __atomic_has_taken_value(c, 0, n);
}
