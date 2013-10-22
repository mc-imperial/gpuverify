//pass
//--local_size=64 --num_groups=64 --clang-opt=-Wno-tautological-compare

bool bar(void);

int baz(void);

__kernel void foo() {
  bool y = bar();
  int x = baz();

  x = !x;
  y = !y;

  if(x){}
  if(!x){}
  if(y){}
  if(!y){}
  if(x < x){}
  if(!(x < x)){}

}
