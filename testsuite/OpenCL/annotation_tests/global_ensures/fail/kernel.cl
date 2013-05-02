//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024



void bar(__local int* A) {
  __requires(!__read(A));
  __global_ensures(!__read(A));
}

__kernel void foo(__local int* A) {

  if((get_local_id(0) % 2) != 0) {
    A[get_local_id(0)] = A[get_local_id(0) + 1];
  }

  if((get_local_id(0) % 2) == 0) {
    bar(A);
  }

}