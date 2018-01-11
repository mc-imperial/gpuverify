//pass
//--local_size=64 --num_groups=64

__kernel void foo(__local int* A, int start, int end) {
    for (unsigned i = start; i < end; ++i) {
      A[0] = 42;
    }
}
