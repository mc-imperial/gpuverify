//pass
//--local_size=64 --num_groups=64



__kernel void foo() {
	
  char h;
  h <<= 1;
  
  h <<= h;
  
  uchar3 v;
  
  v <<= v;
  
  v = v ^ v;
	
}


