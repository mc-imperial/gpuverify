//xfail:BOOGIE_ERROR
//--local_size=1024 --num_groups=1024
//kernel.cl:[\s]+error:[\s]+possible write-write race on \(\(char\*\)p\)\[20]
//kernel.cl:14:6:[\s]+write by thread 5[\s]+in group[\s]+[\d]+[\s]+p\[get_local_id\(0\)] = get_local_id\(0\);
//kernel.cl:17:6:[\s]+write by thread [\d]+[\s]+in group[\s]+[\d]+[\s]+p\[get_local_id\(0\)\+1\] = get_local_id\(0\);




__kernel void foo(__local int* p) {

  for(int i = 0; i < 100; i++) {
		  if(get_local_id(0) == 5) {
			p[get_local_id(0)] = get_local_id(0);
		  }
		  if(get_local_id(0) == 4) {
			p[get_local_id(0)+1] = get_local_id(0);
		  }
  }


}
