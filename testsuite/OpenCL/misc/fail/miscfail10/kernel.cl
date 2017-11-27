//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64
//race

__kernel void foo(__global uint *h1, __global uint *h2, int i)
{
    __global uint *h = i > 0 ? h1 : h2;

    __local uint l[256];

    for(uint i = 0; i < 256; i += 1)
    {
       h[i] += l[i];
    }
}
