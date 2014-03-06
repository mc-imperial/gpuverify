//pass
//--local_size=16 --num_groups=1 --no-inline

void bar(__global int *A) {
  __requires(__enabled());
}

__kernel void foo(__global int* A) {
  bar(A);
}
