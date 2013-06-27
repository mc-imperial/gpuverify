//pass
//--local_size=8 --num_groups=1

#define N 8
__axiom(get_local_size(0) ==  N);

#define tid get_local_id(0)

__kernel void wrap_around() {
  __local int A[N];
  __local int B[N];
  A[tid] = tid;
  __barrier_invariant_2(A[tid] == tid, tid, ((tid+1)%N));
  barrier(CLK_LOCAL_MEM_FENCE);
  B[A[(tid+1)%N]] = tid;
  __assert(B[(tid+1)%N] == tid);
}
