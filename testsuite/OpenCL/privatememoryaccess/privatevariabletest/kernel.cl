//pass
//--local_size=64 --num_groups=64

void bar(int* p) {


}


__kernel void foo() {

  int x;

  x = 10;

  bar(&x);

  int temp = x;

  __assert(temp == 10);

}
