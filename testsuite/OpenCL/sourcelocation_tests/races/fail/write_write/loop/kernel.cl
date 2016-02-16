//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//kernel.cl:[\s]+error:[\s]+possible write-write race on p\[5]
//Write by work item [\d]+ with local id 5 in work group[\s]+[\d]+.+kernel.cl:14:[\d]+:[\s]+p\[get_local_id\(0\)] = get_local_id\(0\);
//Write by work item [\d]+ with local id [\d]+ in work group[\s]+[\d]+.+kernel.cl:17:[\d]+:[\s]+p\[get_local_id\(0\)\+1\] = get_local_id\(0\);




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
