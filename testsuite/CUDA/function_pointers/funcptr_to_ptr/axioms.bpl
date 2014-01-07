axiom (forall x : functionPtr :: {PTR_TO_FUNCPTR(FUNCPTR_TO_PTR(x))} PTR_TO_FUNCPTR(FUNCPTR_TO_PTR(x)) == x);
axiom (forall x : ptr :: {FUNCPTR_TO_PTR(PTR_TO_FUNCPTR(x))} FUNCPTR_TO_PTR(PTR_TO_FUNCPTR(x)) == x);
