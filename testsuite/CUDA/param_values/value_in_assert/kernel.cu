//xfail:NOT_ALL_VERIFIED
//--blockDim=16 --gridDim=16 --no-inline
//a = 12
//b = 36
//c = 48

__global__ void example(unsigned a, unsigned b, unsigned c) {

    __requires(a == 12);
    __requires(b == 36);
    
    __assert(a + b != c);

}
