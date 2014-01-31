using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class LiteralIndexedArrayEliminator
  {
    private GPUVerifier Verifier;

    public LiteralIndexedArrayEliminator(GPUVerifier Verifier)
    {
      this.Verifier = Verifier;
    }

    internal void Eliminate(Program Program)
    {
      var Arrays = CollectRelevantArrays(Program);
      RemoveArraysFromProgram(Program, Arrays);
      ReplaceArraysUsesWithVariables(Program, Arrays);

    }

    private void ReplaceArraysUsesWithVariables(Program Program, Dictionary<string, HashSet<string>> Arrays)
    {
      foreach(var b in Program.Blocks()) {
        b.Cmds = new EliminatorVisitor(Arrays).VisitCmdSeq(b.Cmds);
      }
      foreach(var p in Program.TopLevelDeclarations.OfType<Procedure>()) {
        p.Requires = new EliminatorVisitor(Arrays).VisitRequiresSeq(p.Requires);
        p.Ensures = new EliminatorVisitor(Arrays).VisitEnsuresSeq(p.Ensures);
      }
    }

    private void RemoveArraysFromProgram(Program Program, Dictionary<string, HashSet<string>> Arrays)
    {
      foreach (var a in Verifier.KernelArrayInfo.getPrivateArrays().ToList())
      {
        if (Arrays.ContainsKey(a.Name))
        {
          Verifier.KernelArrayInfo.getPrivateArrays().Remove(a);
          Program.TopLevelDeclarations.Remove(a);

          foreach (var l in Arrays[a.Name])
          {
            Program.TopLevelDeclarations.Add(MakeVariableForArrayIndex(a, l));
          }

        }
      }
    }

    internal static GlobalVariable MakeVariableForArrayIndex(Variable Array, string Literal)
    {
      return new GlobalVariable(
                    Array.tok, new TypedIdent(Array.tok, Array.Name + "$" + Literal,
                      (Array.TypedIdent.Type as MapType).Result));
    }

    private Dictionary<string, HashSet<string>> CollectRelevantArrays(Program Program)
    {
      var Collector = new LiteralIndexVisitor(Verifier);
      Collector.VisitProgram(Program);
      return Collector.LiteralIndexedArrays;
    }

  }

  class LiteralIndexVisitor : StandardVisitor {

    // Maps an array to a set of strings, each of which denotes a literal with which the array can be indexed.
    internal readonly Dictionary<string, HashSet<string>> LiteralIndexedArrays;

    internal LiteralIndexVisitor(GPUVerifier Verifier) {
      this.LiteralIndexedArrays = new Dictionary<string,HashSet<string>>();
      foreach(var v in Verifier.KernelArrayInfo.getPrivateArrays()) {
        this.LiteralIndexedArrays[v.Name] = new HashSet<string>();
      }
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      if(node.Fun is MapSelect && node.Args.Count() == 2) {
        var map = node.Args[0] as IdentifierExpr;
        if(map != null) {
          if(LiteralIndexedArrays.ContainsKey(map.Name)) {
            UpdateIndexingInfo(node.Args[1], map.Name);
          }
        }
      }
      return base.VisitNAryExpr(node);
    }

    public override Cmd VisitAssignCmd(AssignCmd node)
    {
      foreach(var lhs in node.Lhss.OfType<MapAssignLhs>()) {
        if (!(lhs.Map is SimpleAssignLhs)) {
          continue;
        }
        if (lhs.Indexes.Count() != 1) {
          continue;
        }
        var map = (lhs.Map as SimpleAssignLhs).AssignedVariable;
        if(LiteralIndexedArrays.ContainsKey(map.Name)) {
          UpdateIndexingInfo(lhs.Indexes[0], map.Name);
        }
      }
      return base.VisitAssignCmd(node);
    }

    private void UpdateIndexingInfo(Expr MaybeLiteral, string MapName)
    {
      if (MaybeLiteral is LiteralExpr)
      {
        LiteralIndexedArrays[MapName].Add(MaybeLiteral.ToString());
      }
      else
      {
        // The array is not always indexed by a literal
        LiteralIndexedArrays.Remove(MapName);
      }
    }
  }

  class EliminatorVisitor : Duplicator {

    private Dictionary<string, HashSet<string>> Arrays;

    public EliminatorVisitor(Dictionary<string, HashSet<string>> Arrays)
    {
      this.Arrays = Arrays;
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      if(node.Fun is MapSelect && node.Args.Count() == 2) {
        var map = node.Args[0] as IdentifierExpr;
        if(map != null) {
          if(Arrays.ContainsKey(map.Name)) {
            Debug.Assert(node.Args[1] is LiteralExpr);
            return new IdentifierExpr(Token.NoToken,
              LiteralIndexedArrayEliminator.MakeVariableForArrayIndex(map.Decl, node.Args[1].ToString()));
          }
        }
      }
      return base.VisitNAryExpr(node);
    }

    private AssignLhs TransformLhs(AssignLhs lhs) {
      var mapLhs = lhs as MapAssignLhs;
      if (mapLhs == null
        || !(mapLhs.Map is SimpleAssignLhs)
        || mapLhs.Indexes.Count() != 1) {
        return (AssignLhs)Visit(lhs);
      }

      var map = (mapLhs.Map as SimpleAssignLhs).AssignedVariable;

      if(!Arrays.ContainsKey(map.Name)) {
        return (AssignLhs)Visit(lhs);
      }

      Debug.Assert(mapLhs.Indexes[0] is LiteralExpr);

      return new SimpleAssignLhs(
        lhs.tok, new IdentifierExpr(Token.NoToken,
          LiteralIndexedArrayEliminator.MakeVariableForArrayIndex(map.Decl, mapLhs.Indexes[0].ToString())));
    }
    
    public override Cmd VisitAssignCmd(AssignCmd node)
    {
      return new AssignCmd(node.tok, 
        node.Lhss.Select(Item => TransformLhs(Item)).ToList(),
        node.Rhss.Select(Item => VisitExpr(Item)).ToList());
    }

  }

}
