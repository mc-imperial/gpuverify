//pass
//--local_size=32 --num_groups=32 --no-infer

kernel void example(__global int * A) {

    unsigned N = get_global_size(0);
    unsigned tid = get_global_id(0);

    for(int i = 0; i < 100; i++) {
        A[tid] += A[tid + N];
    }

    for(int j = 0; __invariant(j >= 0), j < 100; j++) {
      A[tid + j*N] += A[tid + N];
    }

    __function_wide_invariant(__write_implies(A, ((__write_offset_bytes(A)/sizeof(int))%N) == tid));
    __function_wide_invariant(__read_implies(A, ((__read_offset_bytes(A)/sizeof(int))%N) == tid));

}
