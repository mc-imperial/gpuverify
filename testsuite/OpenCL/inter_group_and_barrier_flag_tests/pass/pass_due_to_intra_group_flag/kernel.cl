//pass
//--local_size=1024 --num_groups=1024 --only-intra-group --no-inline


__kernel void foo(__global int* p) {

  // There would be an inter-group race; this test
  // checks that the flag to check only intra-group
  // races is working
  p[get_local_id(0)] = get_global_id(0);

}
