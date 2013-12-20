//pass
//--local_size=1024 --num_groups=1024 --equality-abstraction --no-inline



__kernel void foo()
{
  __local int A[1024];
  __local int B[1024];

  A[get_local_id(0)] = 0;
  B[get_local_id(0)] = 0;

  for(int i = 0; __invariant(A[get_local_id(0)] == B[get_local_id(0)]), i < 100; i++) {
    A[get_local_id(0)]++;
    B[get_local_id(0)]++;
  }


}
