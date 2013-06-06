__kernel void foo() {
  int i = 0;
  while(
    __assert(i <= 100) // Assertion at loop head treated as invariant
    , // Comma operator separates invariant from loop guard
    i < get_local_id(0) // This is the loop guard
  ) {
    i++;
  }
}