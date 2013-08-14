
function BV32_SGT (bv32, bv32) : bool;

procedure bar(x: int, y: int)
{
  var $$arr : [bv32]int;
  var a : int;
  var b : int;
  var c : int;
  var d : int;

  b := 4; 
  a, b := b, 0;

  $$arr[0bv32] := 12; 
  while (a < 10)
  {
     c := b + 100 - 5;
     a := a + 1;
     if (a == 0 || a == 9)
     {
        d := $$arr[0bv32];
     }
     else
     {
        $$arr[1bv32] := 15; 
        c := x + $$arr[1bv32];
     }
  }
}

procedure {:kernel} foo (x: bv32, y: bv32)
{
  var a : bv32;
  var b : bv32;
  var c : bv32;
  var d : bv32;

  if (BV32_SGT(x, y))
  {
    a := x;
    b := y;
  }
  else
  {
    c := 17bv32;
    a := 2147483647bv32;
    b := 4294967294bv32;
  }
}

