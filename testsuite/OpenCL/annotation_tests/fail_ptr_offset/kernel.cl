//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --no-inline
//kernel.cl:21:3:[\s]+error:[\s]+a precondition for this call might not hold[\s]+bar\(a \+ 1\);
//kernel.cl:10:[\d]+:[\s]+note:[\s]+this is the precondition that might not hold[\s]+__requires\(__ptr_offset_bytes\(p\) == 1\);

void bar(__global int* p) {
  __requires(!__read(p));
  __requires(!__write(p));
  /* Does not hold because offsets are in bytes */
  __requires(__ptr_offset_bytes(p) == 1);

  __global int* q;

  q = p + 1;

  /* Does not hold because offsets are in bytes */
  __assert(__ptr_offset_bytes(q) == 2);
}

__kernel void foo(__global int* a) {
  bar(a + 1);
}

