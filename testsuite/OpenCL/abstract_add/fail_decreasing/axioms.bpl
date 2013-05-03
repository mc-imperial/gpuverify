type T = bv32;

function {:bvbuiltin "bvule"} ULE(T, T) : bool;

axiom (forall x, y : T :: { A(x, y) } ULE(x, A(x, y)) && ULE(y, A(x, y)));