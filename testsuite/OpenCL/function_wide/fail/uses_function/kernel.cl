//xfail:BUGLE_ERROR
//--no-infer --local_size=1024 --num_groups=2 --no-inline

bool bar() {
  return true;
}

kernel void foo() {
    __function_wide_invariant(bar());
}
