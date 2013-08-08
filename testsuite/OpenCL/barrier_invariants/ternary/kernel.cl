//pass
//--local_size=8 --num_groups=1

#define N 8

#define tid get_local_id(0)
#define iseven(x) ((x % 2) == 0)

__kernel void wrap_around() {
  __local int A[N];
  __local int B[N];

  //       0  1  2  3  4  5  6  7
  // A = [ 2, 1, 4, 3, 6, 5, 0, 7 ]
  A[tid] = iseven(tid) ? ((tid+2)%N) : tid;

  __barrier_invariant_2(A[tid] == __ite(iseven(tid), ((tid+2)%N), tid), tid, ((tid+1)%N));
  barrier(CLK_LOCAL_MEM_FENCE);

  //       0  1  2  3  4  5  6  7
  // B = [ 5, 0, 7, 2, 1, 4, 3, 6 ]
  B[A[(tid+1)%N]] = tid;

  // ternary assert okay
  __assert(B[(tid + (iseven(tid) ? 1 : 3))%N] == tid);
  // rephrased assert okay
  __assert(__implies( iseven(tid), B[(tid+1)%N] == tid));
  __assert(__implies(!iseven(tid), B[(tid+3)%N] == tid));
}
