//pass
//--local_size=2048 --num_groups=64

int3 bar(void);

__kernel void foo() {
  int3 x = bar();
}
