type T = bv32;

function {:bvbuiltin "bvule"} ULE(T, T) : bool;

axiom (forall x, y : T :: { A(x, y) } ULE(x, A(x, y)) && ULE(y, A(x, y)));
axiom (forall x : T :: { A(x, 0bv32) } A(x, 0bv32) == x);
axiom (forall x : T :: { A(0bv32, x) } A(0bv32, x) == x);