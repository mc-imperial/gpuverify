__kernel void foo() {
  int j = get_local_id(0) - 1;
  for(int i = get_local_id(0);
      __invariant(i < 200), // First invariant
      __assert(i >= 0),  // Second invariant
      __invariant(j == i - 1), // Third invariant
      i > 0; // Loop guard
      i--
  ) {
    j--;
  }
}