//pass
//--local_size=64 --num_groups=64 --no-inline


int f(int x) {
  return x + 1;
}

__kernel void foo() {

  int y = f(2);

}
