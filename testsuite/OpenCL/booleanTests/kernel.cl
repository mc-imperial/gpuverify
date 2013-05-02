//pass
//--local_size=64 --num_groups=64
__kernel void foo() {
  bool y;
  int x;
  
  x = !x;
  y = !y;
  
  if(x){}
  if(!x){}
  if(y){}
  if(!y){}
  if(x < x){}
  if(!(x < x)){}

}
