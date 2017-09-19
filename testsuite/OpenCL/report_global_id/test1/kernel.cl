//xfail:NOT_ALL_VERIFIED
//--local_size=2,3,4 --num_groups=5,6,7
//Write by work item \(0, 1, 2\) with local id \(0, 1, 2\) in work group \(0, 0, 0\)
//Write by work item \(8, 13, 21\) with local id \(0, 1, 1\) in work group \(4, 4, 5\)

__kernel void foo(global int *p, int x) {
  if(get_global_id(0) == 0 && get_global_id(1) == 1 && get_global_id(2) == 2) {
    p[get_global_id(0)] = get_global_id(1);
  }
  if(get_global_id(0) == 8 && get_global_id(1) == 13 && get_global_id(2) == 21) {
    p[x] = get_global_id(1);
  }
}
