//pass
//--no-infer --local_size=16 --num_groups=16

kernel void foo(__global int* B) {

    int A[] = { 1 };

    B[get_global_id(0)] = A[0];

}
