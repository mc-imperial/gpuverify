//pass
//--blockDim=2048 --gridDim=64

struct s {
  int x;
};

struct t : s {
};

__global__ void foo(t p, t q) {
  p.x = q.x;
}
