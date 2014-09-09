//pass
//--local_size=64 --num_groups=64 --no-annotations

__kernel void foo() {
  for (int i = 0; __invariant(0), i < 5; i++) {}
}
