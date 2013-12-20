//pass
//--local_size=64 --num_groups=64 --no-inline

bool bar(void);

__kernel void foo() {

  bool x = bar();

  x = !x;

}
