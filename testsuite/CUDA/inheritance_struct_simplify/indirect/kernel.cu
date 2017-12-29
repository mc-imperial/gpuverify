//pass
//--blockDim=2048 --gridDim=64

struct s {
  int x;
};

struct t : s {
};

struct u : t {
};

__global__ void foo(u p, u q) {
  p.x = q.x;
}
