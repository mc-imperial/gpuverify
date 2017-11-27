//pass
//--local_size=256 --num_groups=2 --vcgen-opt=/checkArrays:B

kernel void foo(global int * A, global int * B) {
  local int C[256];

  A[get_global_id(0) + 1] = B[get_global_id(0)];
  C[0] = A[get_global_id(0)];
  B[get_global_id(0)] += A[get_global_id(0)];
  __assert(__no_write(A));

}
