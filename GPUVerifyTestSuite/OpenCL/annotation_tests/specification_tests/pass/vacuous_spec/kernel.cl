//pass
//--local_size=1024 --num_groups=1024


// The spec is impossible because no write is made to p

void __spec_bar(__global int* p) {
  __requires(__no_read(p));
  __requires(__write_implies(p, __write_offset(p) == sizeof(int)*get_global_id(0)));
  __requires(p[get_global_id(0)] == get_global_id(0));
  __ensures(p[get_global_id(0)] == __old_int(p[get_global_id(0)] + 1));
}

__kernel void foo(__global int* A) {
  A[get_global_id(0)] = get_global_id(0); 
 __spec_bar(A);
 // Because the spec is impossible it should introduce inconsistency and
 // this should succeed
 __assert(false);
}