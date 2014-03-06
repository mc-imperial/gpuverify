//pass
//--local_size=10 --num_groups=1 --no-inline

int bar(int *x) {
  __requires(x[get_global_id(0)] < 10);
  return x[get_global_id(0)];
}

__kernel void foo(__global int* A) {
  int x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  A[get_global_id(0)] = bar(x);
}
