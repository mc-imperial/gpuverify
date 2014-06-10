//pass
//--local_size=1024 --num_groups=1024 --no-inline


// Purpose of this test is to check a spec which refers to array values

void __spec_bar(__global int* p) {
  __requires(__no_read(p));
  __requires(__write_implies(p, __write_offset_bytes(p)/sizeof(int) == get_global_id(0)));
  __requires(p[get_global_id(0)] == get_global_id(0));
  __ensures(p[get_global_id(0)] == __old_int(p[get_global_id(0)] + 1));
  __ensures(__write_implies(p, __write_offset_bytes(p)/sizeof(int) == get_global_id(0)));
  __writes_to(p);
}

__kernel void foo(__global int* A) {
  A[get_global_id(0)] = get_global_id(0);
 __spec_bar(A);
  __assert(A[get_global_id(0)] == get_global_id(0) + 1);
}
