//pass
//--local_size=32 --num_groups=2

__kernel void foo(__global double *A, int n)
{
  long j = 1;
  for(long i = n; i > 0; i >>= 1) {
    A[get_global_id(0)] = 0.0;
    j *= 2;
  }
}
