type T = bv32;

axiom (forall x, y, z : T :: { A1(x, A(y, z)) } A1(x, A(y, z)) == A1(A(x, y), z));
axiom (forall x, y : T :: { A1(x, y) } A1(x, y) == A(x, y));
axiom (forall x, y, z : T :: { A1(A(x, y), z) } A1(A(x, y), z) == A(A1(x, y), z));
