//pass
//--local_size=16 --num_groups=64 --no-inline


#define MAX 1024


__kernel void arrayCopy(__local float* dst, __constant float* src) {

  int i = get_local_id(0);

  while(
    __invariant(__mod_pow2(i, get_local_size(0)) == get_local_id(0)),
    __invariant(__implies(__write(dst), __mod_pow2(__write_offset(dst), get_local_size(0)*sizeof(float)) == sizeof(float)*get_local_id(0))),
    i < MAX) {
    dst[i] = src[i];
    i += get_local_size(0);

  }


}
