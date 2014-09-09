//pass
//--local_size=64 --num_groups=64 --only-requires

__kernel void foo(global int * A, int j) {
  __requires(j == 0);
  for (int i = 0; __invariant(0), i < 5; i++) {
    A[get_global_id(0) + j] = A[get_global_id(0)];
  }
}
