//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class LiteralIndexedArrayEliminator
  {
    private GPUVerifier Verifier;
    private Dictionary<string, GlobalVariable> arrayCache = new Dictionary<string, GlobalVariable>();

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
      foreach (var b in Program.Blocks()) {
        b.Cmds = new EliminatorVisitor(Arrays, this).VisitCmdSeq(b.Cmds);
      }
      foreach (var p in Program.TopLevelDeclarations.OfType<Procedure>()) {
        p.Requires = new EliminatorVisitor(Arrays, this).VisitRequiresSeq(p.Requires);
        p.Ensures = new EliminatorVisitor(Arrays, this).VisitEnsuresSeq(p.Ensures);
      }
    }

    private void RemoveArraysFromProgram(Program Program, Dictionary<string, HashSet<string>> Arrays)
    {
      foreach (var a in Verifier.KernelArrayInfo.GetPrivateArrays().ToList())
      {
        if (Arrays.ContainsKey(a.Name))
        {
          Verifier.KernelArrayInfo.RemovePrivateArray(a);
          Program.RemoveTopLevelDeclarations(x => x == a);

          foreach (var l in Arrays[a.Name])
          {
            Program.AddTopLevelDeclaration(MakeVariableForArrayIndex(a, l));
          }

        }
      }
    }

    internal GlobalVariable MakeVariableForArrayIndex(Variable Array, string Literal)
    {
      var arrayName = Array.Name + "$" + Literal;
      if (!arrayCache.ContainsKey(arrayName))
      {
        arrayCache[arrayName] = new GlobalVariable(
                    Array.tok, new TypedIdent(Array.tok, arrayName,
                      (Array.TypedIdent.Type as MapType).Result));
      }
      return arrayCache[arrayName];
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
      this.LiteralIndexedArrays = new Dictionary<string, HashSet<string>>();
      foreach (var v in Verifier.KernelArrayInfo.GetPrivateArrays()) {
        this.LiteralIndexedArrays[v.Name] = new HashSet<string>();
      }
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      if (node.Fun is MapSelect && node.Args.Count() == 2) {
        var map = node.Args[0] as IdentifierExpr;
        if (map != null) {
          if (LiteralIndexedArrays.ContainsKey(map.Name)) {
            UpdateIndexingInfo(node.Args[1], map.Name);
          }
        }
      }
      return base.VisitNAryExpr(node);
    }

    public override Cmd VisitAssignCmd(AssignCmd node)
    {
      foreach (var lhs in node.Lhss.OfType<MapAssignLhs>()) {
        if (!(lhs.Map is SimpleAssignLhs)) {
          continue;
        }
        if (lhs.Indexes.Count() != 1) {
          continue;
        }
        var map = (lhs.Map as SimpleAssignLhs).AssignedVariable;
        if (LiteralIndexedArrays.ContainsKey(map.Name)) {
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
    private LiteralIndexedArrayEliminator Eliminator;

    public EliminatorVisitor(Dictionary<string, HashSet<string>> Arrays, LiteralIndexedArrayEliminator Eliminator)
    {
      this.Arrays = Arrays;
      this.Eliminator = Eliminator;
    }

    public override Expr VisitNAryExpr(NAryExpr node)
    {
      if (node.Fun is MapSelect && node.Args.Count() == 2) {
        var map = node.Args[0] as IdentifierExpr;
        if (map != null) {
          if (Arrays.ContainsKey(map.Name)) {
            Debug.Assert(node.Args[1] is LiteralExpr);
            return new IdentifierExpr(Token.NoToken,
              Eliminator.MakeVariableForArrayIndex(map.Decl, node.Args[1].ToString()));
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

      if (!Arrays.ContainsKey(map.Name)) {
        return (AssignLhs)Visit(lhs);
      }

      Debug.Assert(mapLhs.Indexes[0] is LiteralExpr);

      return new SimpleAssignLhs(
        lhs.tok, new IdentifierExpr(Token.NoToken,
          Eliminator.MakeVariableForArrayIndex(map.Decl, mapLhs.Indexes[0].ToString())));
    }

    public override Cmd VisitAssignCmd(AssignCmd node)
    {
      return new AssignCmd(node.tok,
        node.Lhss.Select(Item => TransformLhs(Item)).ToList(),
        node.Rhss.Select(Item => VisitExpr(Item)).ToList());
    }

  }

}
