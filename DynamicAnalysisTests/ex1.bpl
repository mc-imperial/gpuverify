
function BV32_GT (bv32, bv32) : bool;

procedure bar(x: int, y: int)
{
  var $$arr : [bv32]bv32;
  var a : int;
  var b : int;
  
  b := 4; 
  a, b := b, 0;

  $$arr[0bv32] := 12bv32; 
  while (a < 10)
  {
     b := b + 100 - 5;
     a := a + 1;
     if (a == 0 || a == 9)
     {
        b := y;
     }
     else
     {
        b := x;
     }
  }
}

procedure foo (x: bv32, y: bv32)
{
  var a : bv32;
  var b : bv32;

  if (BV32_GT(x, y))
  {
    a := x;
    b := y;
  }
  else
  {
    a := 0bv32;
    b := 1bv32;
  }
}

