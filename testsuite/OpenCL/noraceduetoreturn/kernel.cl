//pass
//--local_size=64 --num_groups=64 --no-inline


int bar(__local float* A)
{

  if(get_local_id(0) != 0) {
    return 0;
  }

  A[4] = 26.8f;

  return 1;

}

__kernel void foo(__local float* A) {

  int y = bar(A);

}
