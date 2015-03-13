//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify {

class VariableDefinitionAnalysisRegion {
  GPUVerifier verifier;
  HashSet<string> possibleInductionVars
      = new HashSet<string>();
  Dictionary<object, Dictionary<string, Expr>> possibleInductionVarDefs
      = new Dictionary<object, Dictionary<string, Expr>>();
  Dictionary<string, Expr> rootSubstitution
      = new Dictionary<string, Expr>();

  VariableDefinitionAnalysisRegion(GPUVerifier v) {
    verifier = v;
  }

  private class MapSelectOrFloatVisitor : StandardVisitor {
    public bool hasMapSelectOrFloat = false;

    private static string[] floatFunctions = new string[]
      { "FADD", "FSUB", "FMUL", "FDIV", "FPOW", "FEQ", "FLT", "FUNO" };

    public override Expr VisitNAryExpr(NAryExpr expr) {
      if (expr.Fun is MapSelect)
        hasMapSelectOrFloat = true;
      else if (floatFunctions.Any(i => expr.Fun.FunctionName.StartsWith(i)))
        hasMapSelectOrFloat = true;
      return base.VisitNAryExpr(expr);
    }
  }

  static private IEnumerable<Tuple<Variable, Expr>> CollectCmds(IEnumerable<Cmd> cmds) {
    foreach (var c in cmds) {
      var aCmd = c as AssignCmd;
      if (aCmd != null) {
        foreach (var a in aCmd.Lhss.Zip(aCmd.Rhss)) {
          var sLhs = a.Item1 as SimpleAssignLhs;
          if (sLhs != null) {
            var v = new MapSelectOrFloatVisitor();
            v.Visit(a.Item2);
            yield return new Tuple<Variable, Expr>(sLhs.DeepAssignedVariable,
                                                   v.hasMapSelectOrFloat ? null : a.Item2);
          }
        }
      }
      var hCmd = c as HavocCmd;
      if (hCmd != null) {
        foreach (var iExpr in hCmd.Vars)
          yield return new Tuple<Variable, Expr>(iExpr.Decl, null);
      }
    }
  }

  private class VarDefs {
    Block block;
    List<Tuple<Variable, Expr>> cmds = new List<Tuple<Variable, Expr>>();
    IEnumerable<Block> predecessors;
    IEnumerable<Variable> modSet;
    Dictionary<Variable, Expr> varDefs = new Dictionary<Variable, Expr>();

    public VarDefs(Block b, IEnumerable<Block> p, IEnumerable<Variable> m) {
      block = b;
      predecessors = p;
      modSet = m;
      cmds = CollectCmds(block.Cmds).ToList();
    }

    private class SubstitutionDuplicator : Duplicator {
      Dictionary<Variable, Expr> defs;
      public bool isSubstitutable = true;

      public SubstitutionDuplicator(Dictionary<Variable, Expr> d) {
        defs = d;
      }

      public override Expr VisitIdentifierExpr(IdentifierExpr expr) {
        if (!defs.ContainsKey(expr.Decl)) {
          return base.VisitIdentifierExpr(expr);
        } else if (defs[expr.Decl] == null) {
          isSubstitutable = false;
          return null;
        } else {
          var dup = new Duplicator();
          return (Expr)dup.Visit(defs[expr.Decl]);
        }
      }
    }

    public void Initialize() {
      foreach (var v in modSet)
        varDefs[v] = Expr.Ident(v);
      foreach (var cmd in cmds) {
        if (cmd.Item2 == null) {
          varDefs[cmd.Item1] = null;
        } else {
          var s = new SubstitutionDuplicator(varDefs);
          var r = (Expr)s.Visit(cmd.Item2);
          varDefs[cmd.Item1] = s.isSubstitutable ? r : null;
        }
      }
    }

    Dictionary<Variable, Expr> MergePredecessors(Dictionary<Block, VarDefs> blockVarDefs,
                                                 IEnumerable<Block> preds) {
      var predVarDefs = new Dictionary<Variable, Expr>();
      foreach (var p in preds) {
        foreach (var varDef in blockVarDefs[p].varDefs)
          if (predVarDefs.ContainsKey(varDef.Key)) {
            if (predVarDefs[varDef.Key] != null &&
                predVarDefs[varDef.Key] != varDef.Value &&
                !predVarDefs[varDef.Key].Equals(varDef.Value))
              predVarDefs[varDef.Key] = null;
          } else {
            predVarDefs[varDef.Key] = varDef.Value;
          }
      }
      return predVarDefs;
    }

    void UpdateAssignment(Variable variable, Expr rhs, Dictionary<Variable, Expr> newVarDefs) {
      if ((varDefs.ContainsKey(variable) && varDefs[variable] == null) ||
          rhs == null) {
        newVarDefs[variable] = null;
        return;
      }

      var s = new SubstitutionDuplicator(newVarDefs);
      var r = (Expr)s.Visit(rhs);
      if (s.isSubstitutable)
        newVarDefs[variable] = r;
      else
        newVarDefs[variable] = null;
    }

    bool HasChanged(Dictionary<Variable, Expr> newVarDefs) {
      if (!varDefs.Any())
        return true;

      var changed = false;
      foreach (var v in newVarDefs) {
        if (varDefs[v.Key] != null && v.Value == null)
          changed = true;
      }
      return changed;
    }

    public bool ComputeTransfer(Dictionary<Block, VarDefs> blockVarDefs) {
      var newVarDefs = MergePredecessors(blockVarDefs, predecessors);
      if (!newVarDefs.Any())
        return false;
      foreach (var cmd in cmds)
        UpdateAssignment(cmd.Item1, cmd.Item2, newVarDefs);
      var changed = HasChanged(newVarDefs);
      varDefs = newVarDefs;
      return changed;
    }

    public IEnumerable<Tuple<Variable, Expr>> FindSelfReferentialVariables(Dictionary<Block, VarDefs> blockVarDefs) {
      var predVarDefs = MergePredecessors(blockVarDefs, predecessors);
      foreach (var p in predVarDefs) {
        if (p.Value == null || p.Value is IdentifierExpr)
          continue;
        var v = new VariablesOccurringInExpressionVisitor();
        v.Visit(p.Value);
        var modVars = v.GetVariables()
                       .Where(i => i is Variable && modSet.Contains(i as Variable));
        if (modVars.Count() == 1 && modVars.Single() as Variable == p.Key)
          yield return new Tuple<Variable, Expr>(p.Key, p.Value);
      }
    }
  }

  void AnalyseRegion(IRegion region, Graph<Block> cfg) {
    var header = region.Header();
    var blockVarDefs = new Dictionary<Block, VarDefs>();
    var blocks = region.SubBlocks().Where(i => i != header);
    var modSet = LoopInvariantGenerator.GetModifiedVariables(region);

    blockVarDefs[header] = new VarDefs(header, cfg.BackEdgeNodes(header), modSet);
    blockVarDefs[header].Initialize();
    foreach (var b in blocks)
      blockVarDefs[b] = new VarDefs(b, cfg.Predecessors(b), modSet);

    var changed = true;
    while (changed) {
      changed = false;
      foreach (var b in blocks)
        if (blockVarDefs[b].ComputeTransfer(blockVarDefs))
          changed = true;
    }

    possibleInductionVarDefs[region.Identifier()]
        = blockVarDefs[header].FindSelfReferentialVariables(blockVarDefs)
                              .ToDictionary(i => i.Item1.Name, i => i.Item2);
  }

  void AnalyseRootRegion(IRegion rootRegion) {
    // We do not track the blocks, as the substitution should still
    // be usable after transformation of the CFG.
    foreach (var cmd in CollectCmds(rootRegion.Cmds())) {
      if (rootSubstitution.ContainsKey(cmd.Item1.Name))
        rootSubstitution[cmd.Item1.Name] = null;
      else
        rootSubstitution[cmd.Item1.Name] = cmd.Item2;
    }
  }

  void Analyse(IRegion rootRegion, Graph<Block> cfg) {
    foreach (var r in rootRegion.SubRegions())
      AnalyseRegion(r, cfg);
    foreach(var v in possibleInductionVarDefs.Select(i => i.Value))
      possibleInductionVars.UnionWith(v.Select(i => i.Key));
    AnalyseRootRegion(rootRegion);
  }

  public Expr GetPossibleInductionVariableDefintion(string variable, object regionId) {
    if (!possibleInductionVarDefs[regionId].ContainsKey(variable))
      return null;
    else
      return possibleInductionVarDefs[regionId][variable];
  }

  public IEnumerable<string> GetPossibleInductionVariables() {
    return possibleInductionVars;
  }

  private class SubstitutionPrimitiveDuplicator : Duplicator {
    Dictionary<string, Expr> defs;

    public SubstitutionPrimitiveDuplicator(Dictionary<string, Expr> d) {
      defs = d;
    }

    public override Expr VisitIdentifierExpr(IdentifierExpr expr) {
      if (!defs.ContainsKey(expr.Name) || defs[expr.Name] == null)
        return base.VisitIdentifierExpr(expr);
      else
        return VisitExpr(defs[expr.Name]);
    }
  }

  public Expr DefOfVariableName(String variable) {
    if (rootSubstitution.ContainsKey(variable) && rootSubstitution[variable] != null) {
      var v = new SubstitutionPrimitiveDuplicator(rootSubstitution);
      return v.VisitExpr(rootSubstitution[variable]);
    } else {
      return null;
    }
  }

  private class SubstitutionDuplicator : Duplicator {
    Dictionary<string, Expr> defs;
    GPUVerifier verifier;
    string procName;
    public HashSet<string> freeVars = new HashSet<string>();

    public SubstitutionDuplicator(Dictionary<string, Expr> d, GPUVerifier v, string p) {
      defs = d;
      verifier = v;
      procName = p;
    }

    public override Expr VisitIdentifierExpr(IdentifierExpr expr) {
      int id;
      var varName = GVUtil.StripThreadIdentifier(expr.Name, out id);
      if (!defs.ContainsKey(varName)) {
        // The variable never assigned to in the procedure
        return base.VisitIdentifierExpr(expr);
      } else if (defs[varName] == null) {
        // The variable has been assigned to, but we do not know what was assigned.
        freeVars.Add(varName);
        return base.VisitIdentifierExpr(expr);
      } else {
        return verifier.MaybeDualise(VisitExpr(defs[varName]), id, procName);
      }
    }
  }

  public Expr SubstDefinitions(Expr expr, string procName, out HashSet<string> freeVars) {
    var v = new SubstitutionDuplicator(rootSubstitution, verifier, procName);
    var r = v.VisitExpr(expr);
    freeVars = v.freeVars;
    return r;
  }

  public Expr SubstDefinitions(Expr expr, string procName) {
    HashSet<string> freeVars;
    return SubstDefinitions(expr, procName, out freeVars);
  }

  public Expr SubstDefinitions(Expr expr, string procName, out bool isConstant, out bool isSubstitutable) {
    HashSet<string> freeVars;
    var r = SubstDefinitions(expr, procName, out freeVars);
    isConstant = !freeVars.Any();
    freeVars.ExceptWith(possibleInductionVars);
    isSubstitutable = !freeVars.Any();
    return r;
  }

  public static VariableDefinitionAnalysisRegion Analyse(Implementation impl, GPUVerifier verifier) {
    var a = new VariableDefinitionAnalysisRegion(verifier);
    a.Analyse(verifier.RootRegion(impl), verifier.Program.ProcessLoops(impl));
    return a;
  }
}

}
