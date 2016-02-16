//xfail:NOT_ALL_VERIFIED
//--local_size=2 --num_groups=5
//Write by work item 0 with local id 0 in work group 0
//Write by work item 7 with local id 1 in work group 3

__kernel void foo(global int *p, int x) {
  if(get_global_id(0) == 0) {
    p[get_global_id(0)] = get_global_id(1);
  }
  if(get_global_id(0) == 7) {
    p[x] = 45;
  }
}
