using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class ArrayBoundsChecker
  {
    private GPUVerifier verifier;
    private Dictionary<string, Tuple<string, int>> ArraySizes;

    private int CurrSourceLocNum = 0;
    private static int ArraySourceID = 0;

    enum BOUND_TYPE {
      LOWER,
      UPPER
    };

    public ArrayBoundsChecker(GPUVerifier verifier, Program program)
    {
      this.verifier = verifier;
      this.ArraySizes = GetBounds(program);
    }

    public Dictionary<string, Tuple<string, int>> GetBounds(Program program)
    {
      Dictionary<string, Tuple<string, int>> arraySizes = new Dictionary<string, Tuple<string, int>>();

      // Get Axioms with array_info attribute, save dictionary of array_info value to (source_name, source_dimensions, ?elem_width, source_elem_width?)
      foreach (Axiom a in program.Axioms.Where(Item => QKeyValue.FindStringAttribute((Item as Axiom).Attributes, "array_info") != null))
      {
        string arrayName = QKeyValue.FindStringAttribute(a.Attributes, "array_info");

        string arraySize = QKeyValue.FindStringAttribute(a.Attributes, "source_dimensions");
        if (arraySize.Equals("*"))
          continue;

        string sourceArrayName = QKeyValue.FindStringAttribute(a.Attributes, "source_name");
        int elemWidth = QKeyValue.FindIntAttribute(a.Attributes, "elem_width", -1);
        int sourceElemWidth = QKeyValue.FindIntAttribute(a.Attributes, "source_elem_width", -1);
        int totalArraySize = ParseArraySize(arraySize) * sourceElemWidth / elemWidth;
        Tuple<string, int> arrayInfo = Tuple.Create<string, int>(sourceArrayName, totalArraySize);
        arraySizes.Add(arrayName, arrayInfo);
      }

      return arraySizes;
    }

    public void CheckBounds(Program program)
    {
      foreach (Implementation i in program.TopLevelDeclarations.ToList().Where(Item => Item.GetType() == typeof(Implementation)))
      {
        foreach (Block b in i.Blocks)
        {
          var newCmds = new List<Cmd>();
          foreach (Cmd c in b.Cmds)
          {
            if (c is AssertCmd && QKeyValue.FindBoolAttribute((c as AssertCmd).Attributes, "sourceloc"))
            {
              CurrSourceLocNum = QKeyValue.FindIntAttribute((c as AssertCmd).Attributes, "sourceloc_num", -1);
            }
            else if (c is AssignCmd)
            {
              List<AccessRecord> accesses = GetArrayAccesses(c as AssignCmd);
              foreach (AccessRecord ar in accesses)
              {
                bool recordedArray = !ArraySizes.ContainsKey(ar.v.Name); /* only gen lower bound check */
                int arrSize = recordedArray ? 0 : ArraySizes[ar.v.Name].Item2;
                newCmds.AddRange(GenArrayBoundChecks(recordedArray, ar, arrSize));
              }
            }
            newCmds.Add(c);
          }
          b.Cmds = newCmds;
        }
      }
    }

    private int ParseArraySize(string arraySize)
    {
      List<string> arrayDimensionSizes = new List<string>(arraySize.Split(','));
      int totalArraySize = 1;
      foreach (string size in arrayDimensionSizes) 
        totalArraySize *= Convert.ToInt32(size);
      return totalArraySize;
    }

    private List<AccessRecord> GetArrayAccesses(AssignCmd assign) 
    {
      List<AccessRecord> accesses = new List<AccessRecord>();
      ReadCollector rc = new ReadCollector(this.verifier.KernelArrayInfo);
      WriteCollector wc = new WriteCollector(this.verifier.KernelArrayInfo);

      foreach (var rhs in assign.Rhss)
      {
        rc.Visit(rhs);
      }
      foreach (AccessRecord ar in rc.nonPrivateAccesses.Union(rc.privateAccesses))
        accesses.Add(ar);

      foreach (var lhs in assign.Lhss)
      {
        wc.Visit(lhs);
      }
      if (wc.FoundNonPrivateWrite() || wc.FoundPrivateWrite())
        accesses.Add(wc.GetAccess());

      return accesses;
    }

    private List<Cmd> GenArrayBoundChecks(bool onlyLower, AccessRecord ar, int arrDim) 
    {
      List<Cmd> boundChecks = new List<Cmd>();

      var ArrayOffset = verifier.FindOrCreateArrayOffsetVariable(ar.v.Name);

      boundChecks.Add(new AssignCmd(Token.NoToken, 
        new List<AssignLhs> { new SimpleAssignLhs(Token.NoToken, Expr.Ident(ArrayOffset)) },
        new List<Expr> { ar.Index }));

      boundChecks.Add(new AssumeCmd(Token.NoToken, Expr.True, 
        new QKeyValue(Token.NoToken, "do_not_predicate", new List<object> { }, 
        new QKeyValue(Token.NoToken, "check_id", new List<object> { "bounds_check_state_" + ArraySourceID },
        new QKeyValue(Token.NoToken, "captureState", new List<object> { "bounds_check_state_" + ArraySourceID }, 
        null)))));
      boundChecks.Add(GenBoundCheck(BOUND_TYPE.LOWER, ar, arrDim, ArrayOffset));

      if (!onlyLower)
        boundChecks.Add(GenBoundCheck(BOUND_TYPE.UPPER, ar, arrDim, ArrayOffset));

      ArraySourceID++;
      return boundChecks;
    }

    private AssertCmd GenBoundCheck(BOUND_TYPE btype, AccessRecord ar, int arrDim, Variable offsetVar)
    {
      Expr boundExpr = null;
      IdentifierExpr offsetVarExpr = new IdentifierExpr(Token.NoToken, offsetVar);
      switch (btype) {
        case BOUND_TYPE.LOWER:
          boundExpr = verifier.IntRep.MakeSge(offsetVarExpr, verifier.IntRep.GetLiteral(0, verifier.size_t_bits)); break;
        case BOUND_TYPE.UPPER:
          boundExpr = verifier.IntRep.MakeSlt(offsetVarExpr, verifier.IntRep.GetLiteral(arrDim, verifier.size_t_bits)); break;
      }

      return new AssertCmd(Token.NoToken, boundExpr,
        new QKeyValue(Token.NoToken, "array_bounds", new List<object> { },
        new QKeyValue(Token.NoToken, "sourceloc_num", new List<object> { new LiteralExpr(Token.NoToken, Microsoft.Basetypes.BigNum.FromInt(CurrSourceLocNum)) },
        new QKeyValue(Token.NoToken, "check_id", new List<object> { "bounds_check_state_" + ArraySourceID },
        new QKeyValue(Token.NoToken, "array_name", new List<object> { ar.v.Name },
        null)))));
    }
  }
}
