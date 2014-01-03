//pass
//--local_size=1024 --num_groups=1024 --no-inline



void bar(__local int* A) {
  __global_requires(!__read(A));
  __global_ensures(!__read(A));
}

__kernel void foo(__local int* A) {

  if((get_local_id(0) % 2) == 0) {
    bar(A);
  }

}
