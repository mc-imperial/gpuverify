//pass
//--local_size=64 --num_groups=64



int bar(int a) {
  __requires(a > 0);
  __ensures(__return_val_int() >= 0);
  return a/2;
}

__kernel void foo() {

  int x;
  x = bar(5);
  __assert(x >= 0);

}


