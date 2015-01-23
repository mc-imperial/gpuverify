//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --equality-abstraction --no-inline
//

//xfail
//--local_size=1024 --num_groups=1024 --equality-abstraction
//

__kernel void foo(__local short* p) {

  p[0]++;

}
