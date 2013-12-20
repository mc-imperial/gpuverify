//pass
//--local_size=64 --num_groups=64 --no-inline

char bar(void);

uchar3 baz(void);

__kernel void foo() {

  char h = bar();
  h <<= 1;

  h <<= h;

  uchar3 v = baz();

  v <<= v;

  v = v ^ v;

}


