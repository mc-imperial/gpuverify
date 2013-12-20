//pass
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* p) {


  for(int i = 0; 
      __candidate_global_invariant(__implies(__write(p), ((__write_offset(p)/sizeof(int)) % get_local_size(0)) == get_local_id(0))),
     i < 100; i++) {
    p[i*get_local_size(0) + get_local_id(0)] = get_local_id(0);
  }
 
}
