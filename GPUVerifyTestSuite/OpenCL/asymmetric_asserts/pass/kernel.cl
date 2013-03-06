//pass
//--local_size=32 --num_groups=32

#define tid get_local_id(0)

__kernel void foo(int x) {
  __requires(x > 32);

  int i = 0;

  while(__invariant(i >= 0), i < (x - (int)tid)) {

    i = i + 1;

    __assert(i >= 0);
    __assert(__other_bool(__implies(__enabled(), i >= 0)));

  }

}