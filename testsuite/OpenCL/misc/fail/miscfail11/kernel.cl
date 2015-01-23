//xfail:NOT_ALL_VERIFIED
//--num_groups=1 --local_size=64 --cruncher-opt=/replaceLoopInvariantAssertions

__kernel void
foo(__global int* A, __global int* fail)
{
    for (int i = 0;
         __invariant(i == 5),
         i < 8; i++) {
    }
}
