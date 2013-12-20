//pass
//--local_size=16 --num_groups=8 --no-inline



__kernel void foo(__global int* A, __global int* B, __global int* C)
{

  int gid = get_global_id(0);

  for(int i = gid;
      __candidate_invariant(i == gid),
          i < 1024; i += 256)
  {
    A[i] = get_local_id(0);
  }

}






/*
int gid$1, gid$2;

int i;

i = 0;

while(i < 1024) {

    if(*) {
        WRITE_HAS_OCCURRED_A = true;
        WRITE_OFFSET_A = i*256 + gid$1;
    }

    assert (!(WRITE_HAS_OCCURRED_A && WRITE_OFFSET_A == i*256 + gid$2));
    assert (!(READ_HAS_OCCURRED_A && READ_OFFSET_A == i*256 + gid$2));

    i = i + 1;
   
 }

*/










/*

phi == (p$1 ==> (WRITE_HAS_OCCURRED_A ==> (WRITE_OFFSET_A % 256) == gid$1));

int gid$1, gid$2;

int i$1, i$2;

i$1 = 0;
i$2 = 0;

p$1 = (i$1 < 1024);
p$2 = (i$2 < 1024);

assert(phi);
havoc(WRITE_HAS_OCCURRED_A, WRITE_OFFSET_A, i$1, i$2, p$1, p$2);
assume(phi);
if(p$1 || p$2) {

    if(p$1) {
        if(*) {
            WRITE_HAS_OCCURRED_A = true;
            WRITE_OFFSET_A = i$1*256 + gid$1;
        }
    }

    assert (!(p$2 && WRITE_HAS_OCCURRED_A && WRITE_OFFSET_A == i$2*256 + gid$2));
    assert (!(p$2 && READ_HAS_OCCURRED_A && READ_OFFSET_A == i$2*256 + gid$2));

    i$1 = p$1 ? i$1 + 1 : i$1;
    i$2 = p$2 ? i$2 + 1 : i$2;

    p$1 = p$1 && (i$1 < 1024);
    p$2 = p$2 && (i$2 < 1024);
    assert(phi);
    assume(false);
 }















modifies(A) = { v : v is assigned to in A }

S;
while(c)
    inv phi;
{
    T;
}
U;


S;
assert(phi);
havoc every variable in modifies(T);
assume(phi);
if(c) {
    T;
    assert(phi);
    assume(false);
}
U;
*/
