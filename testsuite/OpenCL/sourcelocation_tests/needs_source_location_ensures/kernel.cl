//pass
//--local_size=1024 --num_groups=1024 --no-inline



void bar(__global int * p) {
  if(get_global_id(0) == 24) {
    p[0] = get_global_id(0);
  }
}

__kernel void foo(__global int * p) {
 
 bar(p);

 for(int i = 0; i < get_global_id(0); i++) {
 }

}


