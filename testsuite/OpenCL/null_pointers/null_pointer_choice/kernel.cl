//xfail:NOT_ALL_VERIFIED
//--local_size=2 --num_groups=2 --no-inline
//possible null pointer access

__kernel void testKernel() {
  char *bufptr, *next;

  next = NULL;
  bufptr = next + 5;

  if (bufptr >= 0)
    bufptr = NULL;

  bufptr = bufptr;

  bufptr[5] = 'a';
}
