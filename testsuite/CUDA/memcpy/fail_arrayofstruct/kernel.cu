//xfail:NOT_ALL_VERIFIED
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy.
//
//It uses uses struct-assignment, which is translated into a memcpy by clang and
//dealt with as a series of reads/writes by bugle.

typedef struct {
  short x;
  short y;
} pair_t;

__global__ void k(pair_t *pairs) {
  pair_t fresh;
  pairs[42] = fresh;
}
