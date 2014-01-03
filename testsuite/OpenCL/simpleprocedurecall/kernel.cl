//pass
//--local_size=64 --num_groups=64 --no-inline
void bar() {

}

__kernel void foo() {
  bar();
}
