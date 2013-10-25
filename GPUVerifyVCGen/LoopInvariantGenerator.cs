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
using System.Linq;
using System.Text;
using Microsoft.Boogie;
using Microsoft.Basetypes;
using System.Diagnostics;

using GPUVerify.InvariantGenerationRules;

namespace GPUVerify {
  class LoopInvariantGenerator {
    private GPUVerifier verifier;
    private Implementation Impl;

    private List<InvariantGenerationRule> invariantGenerationRules;

    LoopInvariantGenerator(GPUVerifier verifier, Implementation Impl) {
      this.verifier = verifier;
      this.Impl = Impl;

      invariantGenerationRules = new List<InvariantGenerationRule>();
      invariantGenerationRules.Add(new PowerOfTwoInvariantGenerator(verifier));
      invariantGenerationRules.Add(new LoopVariableBoundsInvariantGenerator(verifier));
    }

    public static void PreInstrument(GPUVerifier verifier, Implementation impl) {
      foreach (var region in verifier.RootRegion(impl).SubRegions()) {
        GenerateCandidateForReducedStrengthStrideVariables(verifier, impl, region);
        GenerateCandidateForNonNegativeGuardVariables(verifier, impl, region);
        GenerateCandidateForNonUniformGuardVariables(verifier, impl, region);
      }
    }

    private static void GenerateCandidateForNonUniformGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region) {
        if (!verifier.ContainsBarrierCall(region)) return;

        HashSet<Variable> partitionVars = new HashSet<Variable>();
        HashSet<Variable> guardVars = new HashSet<Variable>();

        foreach (var assume in region.Cmds().OfType<AssumeCmd>().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "partition"))) 
        {
            var visitor = new VariablesOccurringInExpressionVisitor();
            visitor.Visit(assume.Expr);
            partitionVars.UnionWith(visitor.GetVariables());
        }
        var formals = impl.InParams.Select(x => x.Name);
        var modset = GetModifiedVariables(region).Select(x => x.Name);
        foreach (var v in partitionVars)
        {
            Expr expr = verifier.varDefAnalyses[impl].DefOfVariableName(v.Name);
            if (expr == null) continue;
            var visitor = new VariablesOccurringInExpressionVisitor();
            visitor.Visit(expr);
            guardVars.UnionWith(
                visitor.GetVariables().Where(
                  x => x.Name.StartsWith("$") && 
                       !formals.Contains(x.Name) && 
                       modset.Contains(x.Name) &&
                       !verifier.uniformityAnalyser.IsUniform(impl.Name, x.Name) &&
                       x.TypedIdent.Type.Equals(Microsoft.Boogie.Type.GetBvType(32))
                )
            );
        }
        List<AssignCmd> assignments = new List<AssignCmd>();
        foreach (Block b in region.PreHeaders())
        {
            foreach (AssignCmd c in b.Cmds.Where(x => x is AssignCmd)) {
                assignments.Add(c);
            }
        }
        Function otherbv32 = (Function)verifier.ResContext.LookUpProcedure("__other_bv32");
        if (otherbv32 == null) 
        {
            List<Variable> myargs = new List<Variable>();
            myargs.Add(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", Microsoft.Boogie.Type.GetBvType(32))));
            otherbv32 = new Function(Token.NoToken, "__other_bv32", myargs, 
              new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", Microsoft.Boogie.Type.GetBvType(32))));
        }
        foreach (var v in guardVars)
        {
            foreach (AssignCmd c in assignments)
            {
                foreach (var a in c.Lhss.Zip(c.Rhss)) {
                    var lhs = a.Item1;
                    var rhs = a.Item2;
                    if (!(lhs is SimpleAssignLhs)) continue;
                    var sLhs = (SimpleAssignLhs)lhs;
                    var theVar = sLhs.DeepAssignedVariable;
                    if (theVar.Name == v.Name) {
                      var sub = verifier.IntRep.MakeSub(new IdentifierExpr(Token.NoToken, v), rhs as Expr);
                      List<Expr> args = new List<Expr>();
                      args.Add(sub);
                      var inv = Expr.Eq(sub, new NAryExpr(Token.NoToken, new FunctionCall(otherbv32), args));
                      verifier.AddCandidateInvariant(region, inv, "guard minus initial is uniform", InferenceStages.BASIC_CANDIDATE_STAGE);
                    }
                }
            }
        }
    }

    private static void GenerateCandidateForNonNegativeGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region) {
        var visitor = new VariablesOccurringInExpressionVisitor();
        HashSet<Variable> partitionVars = new HashSet<Variable>();
        HashSet<Variable> nonnegVars = new HashSet<Variable>();

        foreach (var assume in region.Cmds().OfType<AssumeCmd>().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "partition"))) 
        {
            visitor.Visit(assume.Expr);
            partitionVars.UnionWith(visitor.GetVariables());
        }
        var formals = impl.InParams.Select(x => x.Name);
        var modset = GetModifiedVariables(region).Select(x => x.Name);
        foreach (var v in partitionVars)
        {
            var expr = verifier.varDefAnalyses[impl].DefOfVariableName(v.Name);
            if (!(expr is NAryExpr)) continue;
            var nary = expr as NAryExpr;
            if (!(nary.Fun.FunctionName.Equals("BV32_SLE") ||
                  nary.Fun.FunctionName.Equals("BV32_SLT") ||
                  nary.Fun.FunctionName.Equals("BV32_SGE") ||
                  nary.Fun.FunctionName.Equals("BV32_SGT"))) continue;
            visitor.Visit(nary);
            nonnegVars.UnionWith(
                visitor.GetVariables().Where(
                  x => x.Name.StartsWith("$") && 
                       !formals.Contains(x.Name) && 
                       modset.Contains(x.Name) &&
                       IsBVType(x.TypedIdent.Type)
                )
            );
        }
        foreach (var v in nonnegVars)
        {
            int BVWidth = (v.TypedIdent.Type as BvType).Bits;
            var inv = verifier.IntRep.MakeSle(verifier.IntRep.GetLiteral(0,BVWidth), new IdentifierExpr(v.tok, v));
            verifier.AddCandidateInvariant(region, inv, "guard variable " + v + " is nonneg", InferenceStages.BASIC_CANDIDATE_STAGE);
        }
    }

    private static bool IsBVType(Microsoft.Boogie.Type type)
    {
        return type.Equals(Microsoft.Boogie.Type.GetBvType(32))
            || type.Equals(Microsoft.Boogie.Type.GetBvType(16))
            || type.Equals(Microsoft.Boogie.Type.GetBvType(8));
    }

    private static void GenerateCandidateForReducedStrengthStrideVariables(GPUVerifier verifier, Implementation impl, IRegion region) {
      var rsa = verifier.reducedStrengthAnalyses[impl];
      foreach (string lc in rsa.StridedLoopCounters(region.Identifier())) {
        var sc = rsa.GetStrideConstraint(lc);
        Variable lcVariable = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, lc,
                Microsoft.Boogie.Type.GetBvType(32)));
        var lcExpr = new IdentifierExpr(Token.NoToken, lcVariable);
        var lcPred = sc.MaybeBuildPredicate(verifier, lcExpr);

        if (lcPred != null) {
          verifier.AddCandidateInvariant(region, lcPred, "variable " + lc + " is strided", InferenceStages.BASIC_CANDIDATE_STAGE);
        }
      }
    }

    public static void PostInstrument(GPUVerifier verifier, Implementation Impl) {
      new LoopInvariantGenerator(verifier, Impl).PostInstrument();
    }

    internal void PostInstrument() {
      HashSet<Variable> LocalVars = new HashSet<Variable>();
      foreach (Variable v in Impl.LocVars) {
        LocalVars.Add(v);
      }
      foreach (Variable v in Impl.InParams) {
        LocalVars.Add(v);
      }
      foreach (Variable v in Impl.OutParams) {
        LocalVars.Add(v);
      }

      AddCandidateInvariants(verifier.RootRegion(Impl), LocalVars, Impl);

    }

    private void AddEqualityCandidateInvariant(IRegion region, string LoopPredicate, Variable v) {
      verifier.AddCandidateInvariant(region,
          Expr.Eq(
              new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable)),
              new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable))
      ), "equality", InferenceStages.BASIC_CANDIDATE_STAGE);
    }

    private void AddPredicatedEqualityCandidateInvariant(IRegion region, string LoopPredicate, Variable v) {
      verifier.AddCandidateInvariant(region, Expr.Imp(
          Expr.And(
              new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$1", Microsoft.Boogie.Type.Int))),
              new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$2", Microsoft.Boogie.Type.Int)))
          ),
          Expr.Eq(
              new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable)),
              new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable))
      )), "predicated equality", InferenceStages.BASIC_CANDIDATE_STAGE);
    }

    private Dictionary<string, int> GetAssignmentCounts(Implementation impl) {

      Dictionary<string, int> result = new Dictionary<string, int>();

      foreach (var c in verifier.RootRegion(impl).Cmds()) {
        if (c is AssignCmd) {
          var aCmd = (AssignCmd)c;
          HashSet<string> alreadySeenInThisAssignment = new HashSet<string>();
          foreach (var a in aCmd.Lhss) {
            if (a is SimpleAssignLhs) {
              var v = GPUVerifier.StripThreadIdentifier(
                ((SimpleAssignLhs)a).AssignedVariable.Name);
              if (!alreadySeenInThisAssignment.Contains(v)) {
                if (result.ContainsKey(v)) {
                  result[v]++;
                }
                else {
                  result[v] = 1;
                }
                alreadySeenInThisAssignment.Add(v);
              }
            }
          }
        }
      }
      return result;
    }


    private void AddBarrierDivergenceCandidates(HashSet<Variable> LocalVars, Implementation Impl, IRegion region)
      {

          if (!verifier.ContainsBarrierCall(region))
          {
              return;
          }

          Expr guard = region.Guard();
          if (guard != null && verifier.uniformityAnalyser.IsUniform(Impl.Name, guard))
          {
              return;
          }

          if (IsDisjunctionOfPredicates(guard))
          {
              string LoopPredicate = ((guard as NAryExpr).Args[0] as IdentifierExpr).Name;
              LoopPredicate = LoopPredicate.Substring(0, LoopPredicate.IndexOf('$'));

              var uniformEnabledPredicate = Expr.Eq(
                  // Int type used here, but it doesn't matter as we will print and then re-parse the program
                  new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$1", Microsoft.Boogie.Type.Int))),
                  new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$2", Microsoft.Boogie.Type.Int)))
              );

              verifier.AddCandidateInvariant(region, uniformEnabledPredicate, "loop predicate equality", InferenceStages.BASIC_CANDIDATE_STAGE);

              verifier.AddCandidateInvariant(region, Expr.Imp(GPUVerifier.ThreadsInSameGroup(), uniformEnabledPredicate), "same group loop predicate equality", InferenceStages.BASIC_CANDIDATE_STAGE);

              Dictionary<string, int> assignmentCounts = GetAssignmentCounts(Impl);

              HashSet<string> alreadyConsidered = new HashSet<String>();  

              foreach (var v in LocalVars)
              {
                string lv = GPUVerifier.StripThreadIdentifier(v.Name);
                if (alreadyConsidered.Contains(lv)) {
                  continue;
                }
                alreadyConsidered.Add(lv);

                if (verifier.uniformityAnalyser.IsUniform(Impl.Name, v.Name))
                {
                    continue;
                }

                if (GPUVerifier.IsPredicate(lv))
                {
                    continue;
                }

                if (!assignmentCounts.ContainsKey(lv) || assignmentCounts[lv] <= 1) {
                  continue;
                }

                if (!verifier.ContainsNamedVariable(
                      GetModifiedVariables(region), lv))
                {
                    continue;
                }

                AddPredicatedEqualityCandidateInvariant(region, LoopPredicate, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, lv, Microsoft.Boogie.Type.Int)));
              }

              if (GPUVerifyVCGenCommandLineOptions.ArrayEqualities)
              {
                  foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays())
                  {
                      if (!verifier.ArrayModelledAdversarially(v))
                      {
                          AddEqualityCandidateInvariant(region, LoopPredicate, v);
                      }
                  }
              }
          }
      }

    private static bool IsDisjunctionOfPredicates(Expr guard) {
      if (!(guard is NAryExpr)) {
        return false;
      }
      NAryExpr nary = (NAryExpr)guard;
      if(nary.Args.Count() != 2) {
        return false;
      }
      if(!(nary.Fun is BinaryOperator)) {
        return false;
      }
      BinaryOperator binOp = (BinaryOperator)nary.Fun;
      if(binOp.Op != BinaryOperator.Opcode.Or) {
        return false;
      }
      if(!(nary.Args[0] is IdentifierExpr && nary.Args[1] is IdentifierExpr)) {
        return false;
      }
      return GPUVerifier.IsPredicate(GPUVerifier.StripThreadIdentifier(
                ((IdentifierExpr)nary.Args[0]).Name)) &&
             GPUVerifier.IsPredicate(GPUVerifier.StripThreadIdentifier(
                ((IdentifierExpr)nary.Args[1]).Name));
    }

    private void AddCandidateInvariants(IRegion region, HashSet<Variable> LocalVars, Implementation Impl) {
      foreach (IRegion subregion in region.SubRegions()) {
        foreach (InvariantGenerationRule r in invariantGenerationRules) {
          r.GenerateCandidates(Impl, subregion);
        }

        AddBarrierDivergenceCandidates(LocalVars, Impl, subregion);

        verifier.RaceInstrumenter.AddRaceCheckingCandidateInvariants(Impl, subregion);

      }
    }

    internal static HashSet<Variable> GetModifiedVariables(IRegion region) {
      HashSet<Variable> result = new HashSet<Variable>();

      foreach (Cmd c in region.Cmds()) {
        List<Variable> vars = new List<Variable>();
        c.AddAssignedVariables(vars);
        foreach (Variable v in vars) {
          result.Add(v);
        }
      }

      return result;
    }

  }
}
