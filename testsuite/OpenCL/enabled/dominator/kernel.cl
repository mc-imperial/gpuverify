//PASS
//--local_size=32 --num_groups=1

__kernel void foo()
{
    int b0 = get_group_id(0);
    int t0 = get_local_id(0);
    __local int A[32];
    int result;

    for (int c2 = b0;
         __global_invariant(__read_implies(A, __read_offset_bytes(A)/sizeof(int) == t0)),
         c2 <  32; c2 += 32) {
      A[t0] = 1;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int c4 = t0;
           __global_invariant(__implies(__dominator_enabled() &__read(A), __read_offset_bytes(A)/sizeof(int) == -t0 + 31)),
           __global_invariant(__write_implies(A, __write_offset_bytes(A)/sizeof(int) == -t0 + 31)),
           c4 < 32; c4 += 16)
        A[-t0 + 31] -= 5;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int c4 = t0; c4 < 32; c4 += 16)
        result = A[t0];
    }
}
