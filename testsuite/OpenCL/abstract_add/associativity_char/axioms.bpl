type T = bv8;

function {:bvbuiltin "bvule"} ULE(T, T) : bool;

axiom (forall x, y, z : T :: { A1(x, A(y, z)) } A1(x, A(y, z)) == A1(A(x, y), z));
axiom (forall x, y : T :: { A1(x, y) } A1(x, y) == A(x, y));
axiom (forall x, y, z : T :: { A1(A(x, y), z) } A1(A(x, y), z) == A(A1(x, y), z));

axiom (forall x, y : T :: { A(x, y) } ULE(x, A(x, y)) && ULE(y, A(x, y)));