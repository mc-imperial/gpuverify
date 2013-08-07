

__kernel void foo() {
    for (int i = get_local_id(0);
         __candidate_invariant(i == 0),
         __candidate_invariant(i >= 0),
         __candidate_invariant(i <= 100),
         i < 100; i+= 10) 
     {

     }

}
