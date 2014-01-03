//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024 --no-inline
//



// Purpose of this test is to check a simple spec

void __spec_bar(__global int* p) {
  __requires(__no_read(p));
  __requires(__no_write(p));
  __reads_from(p);
}

__kernel void foo(__global int* A) {
  __spec_bar(A);
  A[get_global_id(0)] = get_global_id(0);
}
