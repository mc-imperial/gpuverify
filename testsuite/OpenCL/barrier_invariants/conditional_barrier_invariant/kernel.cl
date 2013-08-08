//pass
//--local_size=8 --num_groups=1

#define N 8

#define tid get_local_id(0)
#define iseven(x) ((x % 2) == 0)

__kernel void k() {
  __local int A[N];

  if (iseven(tid)) {
    A[2*tid] = tid;
  } else {
    A[2*tid+1] = tid;
  }

  __barrier_invariant_1(__ite(iseven(tid), A[2*tid] == tid, A[2*tid+1] == tid), tid);
  __barrier_invariant_1(__implies(iseven(tid), A[2*tid] == tid), tid);
  __barrier_invariant_1(__implies(!iseven(tid), A[2*tid+1] == tid), tid);
  barrier(CLK_LOCAL_MEM_FENCE);
}
