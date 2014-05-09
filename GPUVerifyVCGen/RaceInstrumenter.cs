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
using System.IO;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace GPUVerify {

  abstract class RaceInstrumenter : IRaceInstrumenter {

    internal GPUVerifier verifier;

    internal int CheckStateCounter = 0;

    private Dictionary<string, Procedure> RaceCheckingProcedures = new Dictionary<string, Procedure>();

    public RaceInstrumenter(GPUVerifier verifier) {
      this.verifier = verifier;
    }

    private void AddNoAccessCandidateInvariants(IRegion region, Variable v) {

      // Reasoning: if READ_HAS_OCCURRED_v is not in the modifies set for the
      // loop then there is no point adding an invariant
      //
      // If READ_HAS_OCCURRED_v is in the modifies set, but the loop does not
      // contain a barrier, then it is almost certain that a read CAN be
      // pending at the loop head, so the invariant will not hold
      //
      // If there is a barrier in the loop body then READ_HAS_OCCURRED_v will
      // be in the modifies set, but there may not be a live read at the loop
      // head, so it is worth adding the loop invariant candidate.
      //
      // The same reasoning applies for WRITE

      if (verifier.ContainsBarrierCall(region)) {
        foreach (var kind in AccessType.Types)
        {
          if (verifier.ContainsNamedVariable(
              LoopInvariantGenerator.GetModifiedVariables(region), RaceInstrumentationUtil.MakeHasOccurredVariableName(v.Name, kind))) {
            AddNoAccessCandidateInvariant(region, v, kind);
          }
        }
      }
    }

    private void AddNoAccessCandidateRequires(Procedure Proc, Variable v) {
      foreach (var kind in AccessType.Types)
        AddNoAccessCandidateRequires(Proc, v, kind);
    }

    private void AddNoAccessCandidateEnsures(Procedure Proc, Variable v) {
      foreach (var kind in AccessType.Types)
        AddNoAccessCandidateEnsures(Proc, v, kind);
    }

    private void AddNoAccessCandidateInvariant(IRegion region, Variable v, AccessType Access) {
      Expr candidate = NoAccessExpr(v, Access);
      verifier.AddCandidateInvariant(region, candidate, "no" + Access.ToString().ToLower(), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    private void AddSameWarpNoAccessCandidateInvariant(IRegion region, Variable v, AccessType Access) {
      if (!GPUVerifyVCGenCommandLineOptions.WarpSync) return;
      Expr candidate = Expr.Imp(Expr.And(GPUVerifier.ThreadsInSameGroup(), verifier.ThreadsInSameWarp()), NoAccessExpr(v, Access));
      verifier.AddCandidateInvariant(region, candidate, "sameWarpNoAccess", InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE, "do_not_predicate");
    }

    public void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region) {
      List<Expr> offsetPredicatesRead = new List<Expr>();
      List<Expr> offsetPredicatesWrite = new List<Expr>();
      List<Expr> offsetPredicatesAtomic = new List<Expr>();

      foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
        AddNoAccessCandidateInvariants(region, v);
        AddSameWarpNoAccessCandidateInvariant(region, v, AccessType.READ);
        AddSameWarpNoAccessCandidateInvariant(region, v, AccessType.WRITE);
        // Same group and same warp does *not* imply no atomic accesses
        AddOffsetIsBlockBoundedCandidateInvariants(impl, region, v, AccessType.READ);
        AddOffsetIsBlockBoundedCandidateInvariants(impl, region, v, AccessType.WRITE);
        AddOffsetIsBlockBoundedCandidateInvariants(impl, region, v, AccessType.ATOMIC);
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, AccessType.READ);
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, AccessType.WRITE);
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, AccessType.ATOMIC);
        offsetPredicatesRead = CollectOffsetPredicates(impl, region, v, AccessType.READ);
        offsetPredicatesWrite = CollectOffsetPredicates(impl, region, v, AccessType.WRITE);
        offsetPredicatesAtomic = CollectOffsetPredicates(impl, region, v, AccessType.ATOMIC);
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, AccessType.READ, offsetPredicatesRead);
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, AccessType.WRITE, offsetPredicatesWrite);
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, AccessType.ATOMIC, offsetPredicatesAtomic);
        AddComponentBreakingCandidateInvariants(impl, region, v, AccessType.READ);
        AddComponentBreakingCandidateInvariants(impl, region, v, AccessType.WRITE);
        AddComponentBreakingCandidateInvariants(impl, region, v, AccessType.ATOMIC);
      }
    }

    /*
     * Generates candidate invariants by rewriting offset expressions.
     * 
     * A component is an identifier (i.e., local_id_{x,y,z} or group_id_{x,y,z})
     * where /at least one/ component is necessarily distinct between distinct
     * threads. Given an offset expression we extract components using division
     * and subtraction.
     */
    private void AddComponentBreakingCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType Access) {
      // get offset expression
      // continue if there is exactly one offset expression, or,
      // if all offsets are to the same logical element of a vector type (e.g., uint2).
      HashSet<Expr> offsets = GetOffsetsAccessed(region, v, Access);
      if (offsets.Count() == 0) return;
      if (offsets.Count() > 1) {
        HashSet<string> vs = new HashSet<string>();
        foreach (var offset in offsets) {
          var visitor = new VariablesOccurringInExpressionVisitor();
          visitor.Visit(offset);
          vs.UnionWith(visitor.GetVariables().Select(x => x.Name));
          // Could further refine by checking offset has form
          // BV32_ADD(BV32_MUL(v, c+i) where c is a constant 2,3,4
          //                           and   i is in [0,c)
        }
        if (vs.Count() != 1) return;
      }

      // now get the offset definition, distribute and try breaking apart
      bool isConstant;
      var def = verifier.varDefAnalyses[impl].SubstDefinitions(offsets.First(), impl.Name, out isConstant);
      if (def == null || isConstant) return;
      // Should also check expression consists only of adds and muls?
      var distribute = new DistributeExprVisitor(verifier);
      var rewrite = distribute.Visit(def);
      var component = new ComponentVisitor();
      component.Visit(rewrite);
      var invariants = component.GenerateCandidateInvariants(verifier, v, Access);
      foreach (Expr inv in invariants) {
        verifier.AddCandidateInvariant(region, inv, "accessBreak", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
      }
      if (GPUVerifyVCGenCommandLineOptions.ShowAccessBreaking) {
        Console.WriteLine("Access breaking of [{0}]", def);
        component.Dump();
      }
    }

    class FunctionsOccurringInExpressionVisitor : StandardVisitor {
      private HashSet<string> functions = new HashSet<string>();
      internal IEnumerable<string> GetFunctions() {
        return functions;
      }
      public override Expr VisitNAryExpr(NAryExpr node) {
        functions.Add(node.Fun.FunctionName);
        return base.VisitNAryExpr(node);
      }
    }

    class VariablesOrLiteralsOccurringInExpressionVisitor : StandardVisitor {
      private HashSet<Expr> terms = new HashSet<Expr>();
      internal IEnumerable<Expr> GetVariablesOrLiterals() {
        return terms;
      }
      public override Variable VisitVariable(Variable node) {
          terms.Add(new IdentifierExpr(Token.NoToken, node));
          return base.VisitVariable(node);
      }
      public override LiteralExpr VisitLiteralExpr(LiteralExpr node) {
          terms.Add(node);
          return base.VisitLiteralExpr(node);
      }
    }

    /*
     * Generate component candidates from an offset expression.
     *
     * We assume that the offset expression is distributed so that each
     * component appears in a separate multiply subexpression.
     */
    class ComponentVisitor : StandardVisitor {
      private HashSet<string> AllComponents;
      private Dictionary<Expr, HashSet<Expr>> ComponentMap;
      bool CanAccessBreak;

      internal ComponentVisitor() {
        ComponentMap = new Dictionary<Expr, HashSet<Expr>>();
        AllComponents = new HashSet<string>();
        AllComponents.Add((GPUVerifier.MakeThreadId("X", 1)).Name);
        AllComponents.Add((GPUVerifier.MakeThreadId("Y", 1)).Name);
        AllComponents.Add((GPUVerifier.MakeThreadId("Z", 1)).Name);
        AllComponents.Add((GPUVerifier.MakeGroupId("X", 1)).Name);
        AllComponents.Add((GPUVerifier.MakeGroupId("Y", 1)).Name);
        AllComponents.Add((GPUVerifier.MakeGroupId("Z", 1)).Name);
        CanAccessBreak = true;
      }

      public void Dump() {
        if (CanAccessBreak) {
          Console.WriteLine("Can access break expression.");
          foreach (var c in ComponentMap.Keys) {
            var terms = ComponentMap[c];
            Console.WriteLine("Component {0} has {1} multiply terms", c, terms.Count());
            foreach (var t in terms) {
              Console.WriteLine("  Term {0}", t);
            }
          }
        } else {
          Console.WriteLine("Can't access break expression.");
        }
      }

      /*
       * Generate invariants from the populated ComponentMap.
       *
       * For example, the following expression for components c and d:
       *      access = (c * xs) + (d * ys)
       * generates the invariants
       *      c = (access/xs) - (d*ys/xs)
       *      d = (access/ys) - (c*xs/ys)
       */
      public IEnumerable<Expr> GenerateCandidateInvariants(GPUVerifier verifier, Variable v, AccessType Access) {
        if (!CanAccessBreak) return Enumerable.Empty<Expr>();

        var result = new List<Expr>();
        var offsetVar = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits));
        foreach (var c in ComponentMap.Keys) {
          Expr invariant = new IdentifierExpr(Token.NoToken, offsetVar);
          var xs = ComponentMap[c];
          foreach (var x in xs) { 
            invariant = verifier.IntRep.MakeDiv(invariant, x);
          }
          foreach (var d in ComponentMap.Keys.Where(x => x != c)) {
            Expr subexpr = d;
            var ys = ComponentMap[d];
            foreach (var y in ys.Except(xs)) {
              subexpr = verifier.IntRep.MakeMul(subexpr, y);
            }
            foreach (var x in xs.Except(ys)) { 
              subexpr = verifier.IntRep.MakeDiv(subexpr, x);
            }
            invariant = verifier.IntRep.MakeSub(invariant, subexpr);
          }
          invariant = Expr.Eq(c, invariant);
          invariant = Expr.Imp(new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access)), invariant);
          result.Add(invariant);
        }
        return result;
      }

      private bool IsMultiplyExpr(Expr e) {
        var visitor = new FunctionsOccurringInExpressionVisitor();
        visitor.Visit(e);
        var fs = visitor.GetFunctions();
        return (fs.Count() == 1 && fs.Single() == "BV32_MUL");
      }

      /*
       * Extract information for each subexpression containing only multiplies
       * of a single component and terms (literals or variables) and add it to
       * the ComponentMap. We conservatively bail by unsetting CanAccessBreak.
       *
       * TODO: do not ignore guard expression
       * TODO: rewrite for arbitrary bitwidths (currently fixed for BV32)
       */
      public override Expr VisitNAryExpr(NAryExpr node) {
        if (node.Fun.FunctionName.Equals("BV32_MUL")) {
          if (IsMultiplyExpr(node)) {
            var visitor = new VariablesOrLiteralsOccurringInExpressionVisitor();
            visitor.Visit(node);
            var terms = visitor.GetVariablesOrLiterals();
            var components = terms.Where(t => t is IdentifierExpr && AllComponents.Contains((t as IdentifierExpr).Decl.Name));
            if (components.Count() == 0) { // assume guard expression
              return node;
            } else if (components.Count() == 1) {
              if (terms.All(t => (t.Type as BvType).Bits == 32)) {
                var c = components.Single();
                var termsExceptC = new HashSet<Expr>(terms.Where(t => t is LiteralExpr || t != c));
                if (!ComponentMap.ContainsKey(c)) {
                  ComponentMap[c] = termsExceptC;
                  return node;
                }
              }
            }
          }
          // otherwise bail
          CanAccessBreak = false;
          return node;
        }
        return base.VisitNAryExpr(node);
      }
    }

    /*
     * Distribute multiplication over addition in an expression.
     *
     * We rewrite all subexpressions of the form
     *   (mul t (add e1 e2)) 
     * into
     *   (add (mul t e1) (mul t e2))
     *
     * TODO: rewrite for arbitrary bitwidths (currently fixed for BV32)
     */
    class DistributeExprVisitor : Duplicator {
      private GPUVerifier verifier;
      internal DistributeExprVisitor(GPUVerifier verifier) {
        this.verifier = verifier;
      }
      public override Expr VisitNAryExpr(NAryExpr node) {
        if (node.Fun.FunctionName.Equals("BV32_MUL")) {
          Expr lhs = node.Args[0];
          Expr rhs = node.Args[1];
          bool lhsIsLeaf = (lhs is LiteralExpr || lhs is IdentifierExpr);
          bool rhsIsLeaf = (rhs is LiteralExpr || rhs is IdentifierExpr);
          if (!(lhsIsLeaf && rhsIsLeaf)) {
            Expr leaf = lhsIsLeaf ? lhs : rhs;
            NAryExpr term = (lhsIsLeaf ? rhs : lhs) as NAryExpr;
            if (term != null && term.Fun.FunctionName.Equals("BV32_ADD")) {
              Expr e1 = term.Args[0];
              Expr e2 = term.Args[1];
              Expr newLhs =  verifier.IntRep.MakeMul(leaf, e1);
              Expr newRhs =  verifier.IntRep.MakeMul(leaf, e2);
              var visitor = new DistributeExprVisitor(verifier);
              Expr lhs2 = visitor.VisitExpr(newLhs);
              Expr rhs2 = visitor.VisitExpr(newRhs);
              var rewrite = verifier.IntRep.MakeAdd(lhs2, rhs2);
              return base.VisitExpr(rewrite);
            }
          }
        }
        return base.VisitNAryExpr(node);
      }
    }

    private bool DoesNotReferTo(Expr expr, string v) {
      FindReferencesToNamedVariableVisitor visitor = new FindReferencesToNamedVariableVisitor(v);
      visitor.VisitExpr(expr);
      return !visitor.found;
    }

    private int ParameterOffsetForSource(AccessType Access) {
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        return 4;
      }
      else if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.READ) {
        return 3;
      }
      else {
        return 2;
      }
    }

    private List<Expr> CollectOffsetPredicates(Implementation impl, IRegion region, Variable v, AccessType Access) {
      var offsetVar = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits));
      var offsetExpr = new IdentifierExpr(Token.NoToken, offsetVar);
      var offsetPreds = new List<Expr>();

      foreach (var offset in GetOffsetsAccessed(region, v, Access)) {
        bool isConstant;
        var def = verifier.varDefAnalyses[impl].SubstDefinitions(offset, impl.Name, out isConstant);
        if (def == null)
          continue;
        if (isConstant) {
          offsetPreds.Add(Expr.Eq(offsetExpr, def));
        }
        else {
          var sc = StrideConstraint.FromExpr(verifier, impl, def);
          var pred = sc.MaybeBuildPredicate(verifier, offsetExpr);
          if (pred != null)
            offsetPreds.Add(pred);
        }
      }

      return offsetPreds;
    }

    private void AddOffsetIsBlockBoundedCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType Access) {
      var modset = LoopInvariantGenerator.GetModifiedVariables(region).Select(x => x.Name);
      foreach (Expr e in GetOffsetsAccessed(region, v, Access)) {
        if (!(e is NAryExpr)) continue;

        NAryExpr nary = e as NAryExpr;
        if (!nary.Fun.FunctionName.Equals("BV32_ADD")) continue;

        Expr lhs = nary.Args[0];
        Expr rhs = nary.Args[1];
        var lhsVisitor = new VariablesOccurringInExpressionVisitor();
        lhsVisitor.Visit(lhs);
        var lhsVars = lhsVisitor.GetVariables();
        var rhsVisitor = new VariablesOccurringInExpressionVisitor();
        rhsVisitor.Visit(rhs);
        var rhsVars = rhsVisitor.GetVariables();
        Expr constant;
        if (lhsVars.All(x => !modset.Contains(x.Name)) && rhsVars.Any(x =>  modset.Contains(x.Name))) {
          constant = lhs;
        } else if (rhsVars.All(x => !modset.Contains(x.Name)) && lhsVars.Any(x =>  modset.Contains(x.Name))) {
          constant = rhs;
        } else {
          continue;
        }

        Expr lowerBound = verifier.varDefAnalyses[impl].SubstDefinitions(constant, impl.Name);
        if (lowerBound == null) continue;

        var visitor = new VariablesOccurringInExpressionVisitor();
        visitor.VisitExpr(lowerBound);
        var groupIds = visitor.GetVariables().Where(x => GPUVerifier.IsDualisedGroupIdConstant(x));
        if (groupIds.Count() != 1) continue;

        // Getting here means the access consists of a constant (not in the
        // loop's modset) plus a changing index. Furthermore, the constant
        // contains exactly one group-id variable. We guess this forms a lower
        // and upper bound for the access. i.e.,
        //   constant <= access <= constant[group-id+1/group-id]
        Variable groupId = groupIds.Single();
        Expr groupIdPlusOne = verifier.IntRep.MakeAdd(new IdentifierExpr(Token.NoToken, groupId), verifier.IntRep.GetLiteral(1, verifier.size_t_bits));
        Dictionary<Variable, Expr> substs = new Dictionary<Variable, Expr>();
        substs.Add(groupId, groupIdPlusOne);
        Substitution s = Substituter.SubstitutionFromHashtable(substs);
        Expr upperBound = Substituter.Apply(s, lowerBound);
        var lowerBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, Access), verifier.IntRep.MakeSle(lowerBound, OffsetXExpr(v, Access, 1)));
        var upperBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, Access), verifier.IntRep.MakeSlt(            OffsetXExpr(v, Access, 1), upperBound));
        verifier.AddCandidateInvariant(region, lowerBoundInv, "accessLowerBoundBlock", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
        verifier.AddCandidateInvariant(region, upperBoundInv, "accessUpperBoundBlock", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
      }
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType Access) {
      KeyValuePair<IdentifierExpr, Expr> iLessThanC = GetILessThanC(region.Guard(), impl);
      if (iLessThanC.Key != null) {
        foreach (Expr e in GetOffsetsAccessed(region, v, Access)) {
          if (HasFormIPlusLocalIdTimesC(e, iLessThanC, impl)) {
            AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(region, v, iLessThanC.Value, Access);
            break;
          }
        }

        foreach (Expr e in GetOffsetsAccessed(region, v, Access)) {
          if (HasFormIPlusGlobalIdTimesC(e, iLessThanC, impl)) {
            AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(region, v, iLessThanC.Value, Access);
            break;
          }
        }

      }


    }

    private bool HasFormIPlusLocalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl) {
      if (!(e is NAryExpr)) {
        return false;
      }

      NAryExpr nary = e as NAryExpr;

      if (!nary.Fun.FunctionName.Equals("BV32_ADD")) {
        return false;
      }

      return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
          IsLocalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
          (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
          IsLocalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
    }

    private bool IsLocalIdTimesConstant(Expr maybeLocalIdTimesConstant, Expr constant, Implementation impl) {
      if (!(maybeLocalIdTimesConstant is NAryExpr)) {
        return false;
      }
      NAryExpr nary = maybeLocalIdTimesConstant as NAryExpr;
      if (!nary.Fun.FunctionName.Equals("BV32_MUL")) {
        return false;
      }

      return
          (SameConstant(nary.Args[0], constant) && verifier.IsLocalId(nary.Args[1], 0, impl)) ||
          (SameConstant(nary.Args[1], constant) && verifier.IsLocalId(nary.Args[0], 0, impl));
    }


    private bool HasFormIPlusGlobalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl) {
      if (!(e is NAryExpr)) {
        return false;
      }

      NAryExpr nary = e as NAryExpr;

      if (!nary.Fun.FunctionName.Equals("BV32_ADD")) {
        return false;
      }

      return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
          IsGlobalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
          (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
          IsGlobalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
    }

    private bool IsGlobalIdTimesConstant(Expr maybeGlobalIdTimesConstant, Expr constant, Implementation impl) {
      if (!(maybeGlobalIdTimesConstant is NAryExpr)) {
        return false;
      }
      NAryExpr nary = maybeGlobalIdTimesConstant as NAryExpr;
      if (!nary.Fun.FunctionName.Equals("BV32_MUL")) {
        return false;
      }

      return
          (SameConstant(nary.Args[0], constant) && verifier.IsGlobalId(nary.Args[1], 0, impl)) ||
          (SameConstant(nary.Args[1], constant) && verifier.IsGlobalId(nary.Args[0], 0, impl));
    }


    private bool SameConstant(Expr expr, Expr constant) {
      if (constant is IdentifierExpr) {
        IdentifierExpr identifierExpr = constant as IdentifierExpr;
        Debug.Assert(identifierExpr.Decl is Constant);
        return expr is IdentifierExpr && (expr as IdentifierExpr).Decl is Constant && (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Decl.Name);
      }
      else {
        Debug.Assert(constant is LiteralExpr);
        LiteralExpr literalExpr = constant as LiteralExpr;
        if (!(expr is LiteralExpr)) {
          return false;
        }
        if (!(literalExpr.Val is BvConst) || !((expr as LiteralExpr).Val is BvConst)) {
          return false;
        }

        return (literalExpr.Val as BvConst).Value.ToInt == ((expr as LiteralExpr).Val as BvConst).Value.ToInt;
      }
    }

    private bool SameIdentifierExpression(Expr expr, IdentifierExpr identifierExpr) {
      if (!(expr is IdentifierExpr)) {
        return false;
      }
      return (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Name);
    }

    private KeyValuePair<IdentifierExpr, Expr> GetILessThanC(Expr expr, Implementation impl) {

      bool guardHasOuterNot = false;
      if (expr is NAryExpr &&
          (expr as NAryExpr).Fun is BinaryOperator && 
          ((expr as NAryExpr).Fun as BinaryOperator).Op == BinaryOperator.Opcode.And) {
        Expr lhs = (expr as NAryExpr).Args[0];
        Expr rhs = (expr as NAryExpr).Args[1];

        // !v && !v
        if (lhs is NAryExpr && 
              (lhs as NAryExpr).Fun is UnaryOperator &&
              ((lhs as NAryExpr).Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not &&
            rhs is NAryExpr && 
              (rhs as NAryExpr).Fun is UnaryOperator &&
              ((rhs as NAryExpr).Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not) {
          lhs = (lhs as NAryExpr).Args[0];
          rhs = (rhs as NAryExpr).Args[0];
          guardHasOuterNot = true;
        }

        if (lhs is IdentifierExpr && rhs is IdentifierExpr) {
          Variable lhsVar = (lhs as IdentifierExpr).Decl;
          Variable rhsVar = (rhs as IdentifierExpr).Decl;
          if (lhsVar.Name == rhsVar.Name) {
            expr = verifier.varDefAnalyses[impl].DefOfVariableName(lhsVar.Name);
          }
        }
      }

      if (expr is NAryExpr && (expr as NAryExpr).Fun.FunctionName.Equals("bv32_to_bool")) {
        expr = (expr as NAryExpr).Args[0];
      }

      if (!(expr is NAryExpr)) {
        return new KeyValuePair<IdentifierExpr, Expr>(null, null);
      }

      NAryExpr nary = expr as NAryExpr;

      if (!guardHasOuterNot) {
        if (!(nary.Fun.FunctionName.Equals("BV32_C_LT") || 
              nary.Fun.FunctionName.Equals("BV32_LT") || 
              nary.Fun.FunctionName.Equals("BV32_ULT") ||
              nary.Fun.FunctionName.Equals("BV32_SLT")
             )) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        if (!(nary.Args[0] is IdentifierExpr)) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        if (!IsConstant(nary.Args[1])) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        return new KeyValuePair<IdentifierExpr, Expr>(nary.Args[0] as IdentifierExpr, nary.Args[1]);
      } else {
        if (!(nary.Fun.FunctionName.Equals("BV32_C_GT") ||
              nary.Fun.FunctionName.Equals("BV32_GT") ||
              nary.Fun.FunctionName.Equals("BV32_UGT") ||
              nary.Fun.FunctionName.Equals("BV32_SGT")
             )) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        if (!(nary.Args[1] is IdentifierExpr)) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        if (!IsConstant(nary.Args[0])) {
          return new KeyValuePair<IdentifierExpr, Expr>(null, null);
        }

        return new KeyValuePair<IdentifierExpr, Expr>(nary.Args[1] as IdentifierExpr, nary.Args[0]);

      }

    }

    private static bool IsConstant(Expr e) {
      return ((e is IdentifierExpr && (e as IdentifierExpr).Decl is Constant) || e is LiteralExpr);
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateRequires(Procedure Proc, Variable v) {
      foreach (var kind in AccessType.Types)
        AddAccessedOffsetIsThreadLocalIdCandidateRequires(Proc, v, kind);
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(Procedure Proc, Variable v) {
      foreach (var kind in AccessType.Types)
        AddAccessedOffsetIsThreadLocalIdCandidateEnsures(Proc, v, kind);
    }

    public void AddKernelPrecondition() {
      foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
        AddRequiresNoPendingAccess(v);
      }
    }

    public void AddRaceCheckingInstrumentation() {

      foreach (var impl in verifier.Program.Implementations().ToList()) {
        new ImplementationInstrumenter(this, impl).AddRaceCheckCalls();
      }

    }

    protected abstract void AddLogAccessProcedure(Variable v, AccessType Access);

    private void AddRaceCheckingDecsAndProcsForVar(Variable v) {
      foreach (var kind in AccessType.Types)
      {
        AddLogRaceDeclarations(v, kind);
        AddLogAccessProcedure(v, kind);
        AddCheckAccessProcedure(v, kind);
      }
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
        AddUpdateBenignFlagProcedure(v);
      }
    }

    internal Procedure GetRaceCheckingProcedure(IToken tok, string name) {
      if (RaceCheckingProcedures.ContainsKey(name)) {
        return RaceCheckingProcedures[name];
      }
      Procedure newProcedure = new Procedure(tok, name, new List<TypeVariable>(), new List<Variable>(), new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(), new List<Ensures>());
      RaceCheckingProcedures[name] = newProcedure;
      return newProcedure;
    }

    public BigBlock MakeResetReadWriteSetStatements(Variable v, Expr ResetCondition) {
      BigBlock result = new BigBlock(Token.NoToken, null, new List<Cmd>(), null, null);

      foreach (var kind in AccessType.Types)
      {
        Expr ResetAssumeGuard = Expr.Imp(ResetCondition,
          Expr.Not(new IdentifierExpr(Token.NoToken,
            GPUVerifier.MakeAccessHasOccurredVariable(v.Name, kind))));

        if (verifier.KernelArrayInfo.getGlobalArrays().Contains(v))
          ResetAssumeGuard = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), ResetAssumeGuard);

        result.simpleCmds.Add(new AssumeCmd(Token.NoToken, ResetAssumeGuard));
      }
      return result;
    }

    protected Procedure MakeLogAccessProcedureHeader(Variable v, AccessType Access) {
      List<Variable> inParams = new List<Variable>();

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable ValueOldParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value_old", mt.Result));
      Debug.Assert(!(mt.Result is MapType));

      inParams.Add(PredicateParameter);
      inParams.Add(OffsetParameter);
      if(!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        inParams.Add(ValueParameter);
      }
      if(!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        inParams.Add(ValueOldParameter);
      }

      string LogProcedureName = "_LOG_" + Access + "_" + v.Name;

      Procedure result = GetRaceCheckingProcedure(v.tok, LogProcedureName);

      result.InParams = inParams;

      GPUVerifier.AddInlineAttribute(result);

      return result;
    }

    protected Procedure MakeUpdateBenignFlagProcedureHeader(Variable v) {
      List<Variable> inParams = new List<Variable>();

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Debug.Assert(!(mt.Result is MapType));

      inParams.Add(PredicateParameter);
      inParams.Add(OffsetParameter);

      string UpdateBenignFlagProcedureName = "_UPDATE_WRITE_READ_BENIGN_FLAG_" + v.Name;

      Procedure result = GetRaceCheckingProcedure(v.tok, UpdateBenignFlagProcedureName);

      result.InParams = inParams;

      GPUVerifier.AddInlineAttribute(result);

      return result;
    }

    protected Procedure MakeCheckAccessProcedureHeader(Variable v, AccessType Access) {
      List<Variable> inParams = new List<Variable>();

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Debug.Assert(!(mt.Result is MapType));

      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

      inParams.Add(PredicateParameter);
      inParams.Add(OffsetParameter);
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        inParams.Add(ValueParameter);
      }

      string CheckProcedureName = "_CHECK_" + Access + "_" + v.Name;

      Procedure result = GetRaceCheckingProcedure(v.tok, CheckProcedureName);

      result.InParams = inParams;

      return result;
    }

    public void AddRaceCheckingCandidateRequires(Procedure Proc) {
      foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
        AddNoAccessCandidateRequires(Proc, v);
        AddReadOrWrittenOffsetIsThreadIdCandidateRequires(Proc, v);
      }
    }

    public void AddRaceCheckingCandidateEnsures(Procedure Proc) {
      foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
        AddNoAccessCandidateEnsures(Proc, v);
        AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(Proc, v);
      }
    }

    private void AddNoAccessCandidateRequires(Procedure Proc, Variable v, AccessType Access) {
      verifier.AddCandidateRequires(Proc, NoAccessExpr(v, Access), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    private void AddNoAccessCandidateEnsures(Procedure Proc, Variable v, AccessType Access) {
      verifier.AddCandidateEnsures(Proc, NoAccessExpr(v, Access), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    private HashSet<Expr> GetOffsetsAccessed(IRegion region, Variable v, AccessType Access) {
      HashSet<Expr> result = new HashSet<Expr>();

      foreach (Cmd c in region.Cmds()) {
        if (c is CallCmd) {
          CallCmd call = c as CallCmd;

          if (call.callee == "_LOG_" + Access + "_" + v.Name) {
            // Ins[0] is thread 1's predicate,
            // Ins[1] is the offset to be read
            // If Ins[1] has the form BV32_ADD(offset#construct...(P), offset),
            // we are looking for the second parameter to this BV32_ADD
            Expr offset = call.Ins[1];
            if (offset is NAryExpr) {
              var nExpr = (NAryExpr)offset;
              if (nExpr.Fun.FunctionName == "BV32_ADD" &&
                  nExpr.Args[0] is NAryExpr) {
                var n0Expr = (NAryExpr)nExpr.Args[0];
                if (n0Expr.Fun.FunctionName.StartsWith("offset#"))
                  offset = nExpr.Args[1];
              }
            }
            result.Add(offset);
          }

        }

      }

      return result;
    }

    public virtual void AddRaceCheckingDeclarations() {
      foreach (Variable v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
        AddRaceCheckingDecsAndProcsForVar(v);
      }
    }

    protected void AddUpdateBenignFlagProcedure(Variable v) {
      Procedure UpdateBenignFlagProcedure = MakeUpdateBenignFlagProcedureHeader(v);

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      Variable AccessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE);
      Variable AccessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, AccessType.WRITE, verifier.IntRep.GetIntType(verifier.size_t_bits));
      Variable AccessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));

      Debug.Assert(!(mt.Result is MapType));

      List<Variable> locals = new List<Variable>();
      List<BigBlock> bigblocks = new List<BigBlock>();
      List<Cmd> simpleCmds = new List<Cmd>();

      Expr Condition = Expr.And(new IdentifierExpr(v.tok, PredicateParameter),
                         Expr.And(new IdentifierExpr(v.tok, AccessHasOccurredVariable),
                           Expr.Eq(new IdentifierExpr(v.tok, AccessOffsetVariable),
                             new IdentifierExpr(v.tok, OffsetParameter))));

        simpleCmds.Add(MakeConditionalAssignment(AccessBenignFlagVariable,
            Condition, Expr.False));

        bigblocks.Add(new BigBlock(v.tok, "_UPDATE_BENIGN_FLAG", simpleCmds, null, null));

        Implementation UpdateBenignFlagImplementation = new Implementation(v.tok, "_UPDATE_WRITE_READ_BENIGN_FLAG_" + v.Name, new List<TypeVariable>(), UpdateBenignFlagProcedure.InParams, new List<Variable>(), locals, new StmtList(bigblocks, v.tok));
        GPUVerifier.AddInlineAttribute(UpdateBenignFlagImplementation);

        UpdateBenignFlagImplementation.Proc = UpdateBenignFlagProcedure;

        verifier.Program.TopLevelDeclarations.Add(UpdateBenignFlagProcedure);
        verifier.Program.TopLevelDeclarations.Add(UpdateBenignFlagImplementation);
    }

    protected void AddCheckAccessProcedure(Variable v, AccessType Access) {
      Procedure CheckAccessProcedure = MakeCheckAccessProcedureHeader(v, Access);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Debug.Assert(!(mt.Result is MapType));

      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));

      if (Access == AccessType.READ) {
        Variable WriteReadBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

        Expr NoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          NoBenignTest = new IdentifierExpr(Token.NoToken, WriteReadBenignFlagVariable);
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, NoBenignTest, AccessType.WRITE, "write_read");

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.ATOMIC, "atomic_read");
        }
      }
      else if (Access == AccessType.WRITE) {
        Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

        Expr WriteNoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          WriteNoBenignTest = Expr.Neq(
              new IdentifierExpr(Token.NoToken, RaceInstrumentationUtil.MakeValueVariable(v.Name, AccessType.WRITE, mt.Result)),
              new IdentifierExpr(Token.NoToken, ValueParameter));
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, WriteNoBenignTest, AccessType.WRITE, "write_write");

        Expr ReadNoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          ReadNoBenignTest = Expr.Neq(
              new IdentifierExpr(Token.NoToken, RaceInstrumentationUtil.MakeValueVariable(v.Name, AccessType.READ, mt.Result)),
              new IdentifierExpr(Token.NoToken, ValueParameter));
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, ReadNoBenignTest, AccessType.READ, "read_write");

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.ATOMIC, "atomic_write");
        }
      }
      else if (Access == AccessType.ATOMIC) {
        if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.WRITE, "write_atomic");
        }

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.READ, "read_atomic");
        }
      }

      verifier.Program.TopLevelDeclarations.Add(CheckAccessProcedure);
    }

    protected void AddCheckAccessCheck(Variable v, Procedure CheckAccessProcedure, Variable PredicateParameter, Variable OffsetParameter, Expr NoBenignTest, AccessType Access, String attribute) {
      // Check atomic by thread 2 does not conflict with read by thread 1
      Variable AccessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      Variable AccessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits));

      Expr AccessGuard = new IdentifierExpr(Token.NoToken, PredicateParameter);
      AccessGuard = Expr.And(AccessGuard, new IdentifierExpr(Token.NoToken, AccessHasOccurredVariable));
      AccessGuard = Expr.And(AccessGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, AccessOffsetVariable),
                                new IdentifierExpr(Token.NoToken, OffsetParameter)));

      if (NoBenignTest != null) {
        AccessGuard = Expr.And(AccessGuard, NoBenignTest);
      }

      if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
        AccessGuard = Expr.And(AccessGuard, GPUVerifier.ThreadsInSameGroup());
      }

      AccessGuard = Expr.Not(AccessGuard);

      Requires NoAccessRaceRequires = new Requires(false, AccessGuard);

      string originalName = verifier.GlobalArrayOriginalNames[v.Name];
      Debug.Assert(originalName != null);

      NoAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, attribute, new List<object>(), null);
      NoAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoAccessRaceRequires.Attributes);
      NoAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoAccessRaceRequires.Attributes);
      NoAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "original_name", new List<object>() { originalName }, NoAccessRaceRequires.Attributes);
      CheckAccessProcedure.Requires.Add(NoAccessRaceRequires);
    }

    protected void AddLogRaceDeclarations(Variable v, AccessType Access) {
      verifier.FindOrCreateAccessHasOccurredVariable(v.Name, Access);
      verifier.FindOrCreateOffsetVariable(v.Name, Access);

      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        Debug.Assert(v.TypedIdent.Type is MapType);
        MapType mt = v.TypedIdent.Type as MapType;
        Debug.Assert(mt.Arguments.Count == 1);
        verifier.FindOrCreateValueVariable(v.Name, Access, mt.Result);
      }

      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        Debug.Assert(v.TypedIdent.Type is MapType);
        MapType mt = v.TypedIdent.Type as MapType;
        Debug.Assert(mt.Arguments.Count == 1);
        verifier.FindOrCreateBenignFlagVariable(v.Name);
      }

      if ((Access == AccessType.READ || Access == AccessType.WRITE) &&
           verifier.ArraysAccessedByAsyncWorkGroupCopy[Access].Contains(v.Name)) {
        verifier.FindOrCreateAsyncHandleVariable(v.Name, Access);
      }

    }


    protected static AssignCmd MakeConditionalAssignment(Variable lhs, Expr condition, Expr rhs) {
      List<AssignLhs> lhss = new List<AssignLhs>();
      List<Expr> rhss = new List<Expr>();
      lhss.Add(new SimpleAssignLhs(lhs.tok, new IdentifierExpr(lhs.tok, lhs)));
      rhss.Add(new NAryExpr(rhs.tok, new IfThenElse(rhs.tok), new List<Expr>(new Expr[] { condition, rhs, new IdentifierExpr(lhs.tok, lhs) })));
      return new AssignCmd(lhs.tok, lhss, rhss);
    }

    private Expr MakeAccessedIndex(Variable v, Expr offsetExpr, AccessType Access) {
      Expr result = new IdentifierExpr(v.tok, v.Clone() as Variable);
      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      result = Expr.Select(result,
          new Expr[] { offsetExpr });
      Debug.Assert(!(mt.Result is MapType));
      return result;
    }

    protected void AddRequiresNoPendingAccess(Variable v) {
      IdentifierExpr ReadAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.READ));
      IdentifierExpr WriteAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE));
      IdentifierExpr AtomicAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.ATOMIC));

      foreach (var Proc in verifier.KernelProcedures.Keys) {
        Proc.Requires.Add(new Requires(false,Expr.And(Expr.And(Expr.Not(ReadAccessOccurred1), Expr.Not(WriteAccessOccurred1)),Expr.Not(AtomicAccessOccurred1))));
      }
    }

    private Expr BuildAccessOccurredFalseExpr(string name, AccessType Access)
    {
      return Expr.Imp(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(name, Access)),
                                         Expr.False);
    }

    private AssertCmd BuildAccessOccurredFalseInvariant(string name, AccessType Access)
    {
      return new AssertCmd(Token.NoToken, BuildAccessOccurredFalseExpr(name, Access));
    }

    protected Expr NoAccessExpr(Variable v, AccessType Access) {
      Variable AccessHasOccurred = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      Expr expr = Expr.Not(new IdentifierExpr(v.tok, AccessHasOccurred));
      return expr;
    }


    protected void AddOffsetsSatisfyPredicatesCandidateInvariant(IRegion region, Variable v, AccessType Access, List<Expr> preds) {
      if (preds.Count != 0) {
        Expr expr = AccessedOffsetsSatisfyPredicatesExpr(v, preds, Access);
        verifier.AddCandidateInvariant(region, expr, "accessedOffsetsSatisfyPredicates", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
      }
    }

    private Expr AccessedOffsetsSatisfyPredicatesExpr(Variable v, IEnumerable<Expr> offsets, AccessType Access) {
      return Expr.Imp(
              new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access)),
              offsets.Aggregate(Expr.Or));
    }

    private Expr AccessedOffsetIsThreadLocalIdExpr(Variable v, AccessType Access) {
      Expr offsetVar = new IdentifierExpr(v.tok, RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits)));
      Expr offsetExpr = verifier.IntRep.MakeZext(new IdentifierExpr(v.tok, GPUVerifier.MakeThreadId("X", 1)), offsetVar.Type);
      return Expr.Imp(
                new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access)),
                Expr.Eq(offsetVar, offsetExpr));
    }

    private Expr GlobalIdExpr(string dimension, int Thread) {
      return new VariableDualiser(Thread, null, null).VisitExpr(verifier.GlobalIdExpr(dimension).Clone() as Expr);
    }

    protected void AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(IRegion region, Variable v, Expr constant, AccessType Access) {
      Expr expr = MakeCTimesLocalIdRangeExpression(v, constant, Access);
      verifier.AddCandidateInvariant(region,
          expr, "accessedOffsetInRangeCTimesLid", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    private Expr MakeCTimesLocalIdRangeExpression(Variable v, Expr constant, AccessType Access) {
      Expr CTimesLocalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
          new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", 1)));

      Expr CTimesLocalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
          new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", 1))), constant.Clone() as Expr);

      Expr CTimesLocalIdLeqAccessedOffset = verifier.IntRep.MakeSle(CTimesLocalId, OffsetXExpr(v, Access, 1));

      Expr AccessedOffsetLtCTimesLocalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, Access, 1), CTimesLocalIdPlusC);

      return Expr.Imp(
              GPUVerifier.MakeAccessHasOccurredExpr(v.Name, Access),
              Expr.And(CTimesLocalIdLeqAccessedOffset, AccessedOffsetLtCTimesLocalIdPlusC));
    }

    private IdentifierExpr OffsetXExpr(Variable v, AccessType Access, int Thread) {
      return new IdentifierExpr(v.tok, new VariableDualiser(Thread, null, null).VisitVariable(RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits))));
    }

    protected void AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(IRegion region, Variable v, Expr constant, AccessType Access) {
      Expr expr = MakeCTimesGloalIdRangeExpr(v, constant, Access);
      verifier.AddCandidateInvariant(region,
          expr, "accessedOffsetInRangeCTimesGid", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    private Expr MakeCTimesGloalIdRangeExpr(Variable v, Expr constant, AccessType Access) {
      Expr CTimesGlobalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
          GlobalIdExpr("X", 1));

      Expr CTimesGlobalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
          GlobalIdExpr("X", 1)), constant.Clone() as Expr);

      Expr CTimesGlobalIdLeqAccessedOffset = verifier.IntRep.MakeSle(CTimesGlobalId, OffsetXExpr(v, Access, 1));

      Expr AccessedOffsetLtCTimesGlobalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, Access, 1), CTimesGlobalIdPlusC);

      Expr implication = Expr.Imp(
              GPUVerifier.MakeAccessHasOccurredExpr(v.Name, Access),
              Expr.And(CTimesGlobalIdLeqAccessedOffset, AccessedOffsetLtCTimesGlobalIdPlusC));
      return implication;
    }

    protected void AddAccessedOffsetIsThreadLocalIdCandidateRequires(Procedure Proc, Variable v, AccessType Access) {
      verifier.AddCandidateRequires(Proc, AccessedOffsetIsThreadLocalIdExpr(v, Access), InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    protected void AddAccessedOffsetIsThreadLocalIdCandidateEnsures(Procedure Proc, Variable v, AccessType Access) {
      verifier.AddCandidateEnsures(Proc, AccessedOffsetIsThreadLocalIdExpr(v, Access), InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

  }



  class FindReferencesToNamedVariableVisitor : StandardVisitor {
    internal bool found = false;
    private string name;

    internal FindReferencesToNamedVariableVisitor(string name) {
      this.name = name;
    }

    public override Variable VisitVariable(Variable node) {
      if (GVUtil.StripThreadIdentifier(node.Name).Equals(name)) {
        found = true;
      }
      return base.VisitVariable(node);
    }
  }

  class ImplementationInstrumenter {

    private RaceInstrumenter RI;
    private GPUVerifier verifier;
    private Implementation impl;
    private QKeyValue SourceLocationAttributes = null;
    private int AsyncIndexTempCounter = 0;

    internal ImplementationInstrumenter(RaceInstrumenter RI, Implementation impl) {
      this.RI = RI;
      this.verifier = RI.verifier;
      this.impl = impl;
    }

    internal void AddRaceCheckCalls() {
      impl.Blocks = impl.Blocks.Select(AddRaceCheckCalls).ToList();
    }

    private void AddRaceCheckCalls(Implementation impl) {
    }

    private Block AddRaceCheckCalls(Block b) {
      b.Cmds = AddRaceCheckCalls(b.Cmds);
      return b;
    }

    private List<Cmd> AddRaceCheckCalls(List<Cmd> cs) {
      var result = new List<Cmd>();
      foreach (Cmd c in cs) {

        if (c is AssertCmd) {
          AssertCmd assertion = c as AssertCmd;
          if (QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc")) {
            SourceLocationAttributes = assertion.Attributes;
            // Remove source location assertions
            continue;
          }
        }

        if (c is CallCmd) {
          CallCmd call = c as CallCmd;
          if (QKeyValue.FindBoolAttribute(call.Attributes,"atomic"))
          {
            AddLogAndCheckCalls(result,new AccessRecord((call.Ins[0] as IdentifierExpr).Decl,call.Ins[1]),AccessType.ATOMIC,null);
            if (!GPUVerifyVCGenCommandLineOptions.OnlyWarp)
            {
              if (!GPUVerifyVCGenCommandLineOptions.OnlyLog) {
                (result[result.Count() - 1] as CallCmd).Attributes.AddLast((QKeyValue) call.Attributes.Clone()); // Magic numbers ahoy! -1 should be the check
              }
              int logOffset = GPUVerifyVCGenCommandLineOptions.OnlyLog ? 1 : 3;
              (result[result.Count() - logOffset] as CallCmd).Attributes.AddLast((QKeyValue) call.Attributes.Clone()); // And -logOffset should be the log
            }
            Debug.Assert(call.Outs.Count() == 2); // The receiving variable and the array should be assigned to
            result.Add(new HavocCmd(Token.NoToken, new List<IdentifierExpr> { call.Outs[0] })); // We havoc the receiving variable.  We do not need to havoc the array, because it *must* be the case that this array is modelled adversarially
            continue;
          }

          if (QKeyValue.FindBoolAttribute(call.Attributes, "async_work_group_copy"))
          {
            IdentifierExpr DstArray = (IdentifierExpr)call.Ins[0];
            Expr DstOffset = call.Ins[1];
            IdentifierExpr SrcArray = (IdentifierExpr)call.Ins[2];
            Expr SrcOffset = call.Ins[3];
            Expr Size = call.Ins[4];
            Expr Handle = call.Ins[5];

            IdentifierExpr IndexTemp = new IdentifierExpr(Token.NoToken,
              new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_async_index_temp_" + AsyncIndexTempCounter, Size.Type)));
            AsyncIndexTempCounter++;

            impl.LocVars.Add(IndexTemp.Decl);

            result.Add(new HavocCmd(Token.NoToken, new List<IdentifierExpr> { IndexTemp }));
            result.Add(new AssumeCmd(Token.NoToken, verifier.IntRep.MakeUge(IndexTemp, verifier.IntRep.GetLiteral(0, verifier.size_t_bits))));
            result.Add(new AssumeCmd(Token.NoToken, verifier.IntRep.MakeUlt(IndexTemp, Size)));

            AddLogAndCheckCalls(result, 
              new AccessRecord(DstArray.Decl, verifier.IntRep.MakeAdd(DstOffset, IndexTemp)),
              AccessType.WRITE,
              null);
            AddLogAndCheckCalls(result, 
              new AccessRecord(SrcArray.Decl, verifier.IntRep.MakeAdd(SrcOffset, IndexTemp)),
              AccessType.WRITE,
              null);

            continue;

          }
        }

        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
          foreach (var rhs in assign.Rhss)
            rc.Visit(rhs);
          if (rc.accesses.Count > 0) {
            foreach (AccessRecord ar in rc.accesses) {
              if(!verifier.KernelArrayInfo.getReadOnlyNonLocalArrays().Contains(ar.v)) {
                AddLogAndCheckCalls(result, ar, AccessType.READ, null);
              }
            }
          }

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            WriteCollector wc = new WriteCollector(verifier.KernelArrayInfo);
            wc.Visit(LhsRhs.Item1);
            if (wc.FoundWrite()) {
              AccessRecord ar = wc.GetAccess();
              AddLogAndCheckCalls(result, ar, AccessType.WRITE, LhsRhs.Item2);
            }
          }
        }

        result.Add(c);

      }
      return result;
    }

    private void AddLogAndCheckCalls(List<Cmd> result, AccessRecord ar, AccessType Access, Expr Value) {
      if (!GPUVerifyVCGenCommandLineOptions.OnlyWarp || Access == AccessType.WRITE) {
        result.Add(MakeLogCall(ar, Access, Value));
        if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
          result.Add(MakeUpdateBenignFlagCall(ar));
        }
        if (!GPUVerifyVCGenCommandLineOptions.OnlyLog) {
          result.Add(MakeCheckCall(result, ar, Access, Value));
        }
      }
    }

    private CallCmd MakeCheckCall(List<Cmd> result, AccessRecord ar, AccessType Access, Expr Value) {
      if(SourceLocationAttributes == null) {
        ExitWithNoSourceError(ar.v, Access);
      }
      List<Expr> inParamsChk = new List<Expr>();
      inParamsChk.Add(ar.Index);
      MaybeAddValueParameter(inParamsChk, ar, Value, Access);
      Procedure checkProcedure = RI.GetRaceCheckingProcedure(Token.NoToken, "_CHECK_" + Access + "_" + ar.v.Name);
      verifier.OnlyThread2.Add(checkProcedure.Name);
      string CheckState = "check_state_" + RI.CheckStateCounter;
      RI.CheckStateCounter++;
      AssumeCmd captureStateAssume = new AssumeCmd(Token.NoToken, Expr.True);
      captureStateAssume.Attributes = SourceLocationAttributes.Clone() as QKeyValue;
      captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
        "captureState", new List<object>() { CheckState }, captureStateAssume.Attributes);
      captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
        "check_id", new List<object>() { CheckState }, captureStateAssume.Attributes);
      captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
        "do_not_predicate", new List<object>() { }, captureStateAssume.Attributes);
      
      result.Add(captureStateAssume);
      CallCmd checkAccessCallCmd = new CallCmd(Token.NoToken, checkProcedure.Name, inParamsChk, new List<IdentifierExpr>());
      checkAccessCallCmd.Proc = checkProcedure;
      checkAccessCallCmd.Attributes = SourceLocationAttributes.Clone() as QKeyValue;
      checkAccessCallCmd.Attributes = new QKeyValue(Token.NoToken, "check_id", new List<object>() { CheckState }, checkAccessCallCmd.Attributes);
      return checkAccessCallCmd;
    }

    private CallCmd MakeLogCall(AccessRecord ar, AccessType Access, Expr Value) {
      if(SourceLocationAttributes == null) {
        ExitWithNoSourceError(ar.v, Access);
      }
      List<Expr> inParamsLog = new List<Expr>();
      inParamsLog.Add(ar.Index);
      MaybeAddValueParameter(inParamsLog, ar, Value, Access);
      MaybeAddValueOldParameter(inParamsLog, ar, Access);
      Procedure logProcedure = RI.GetRaceCheckingProcedure(Token.NoToken, "_LOG_" + Access + "_" + ar.v.Name);
      verifier.OnlyThread1.Add(logProcedure.Name);
      CallCmd logAccessCallCmd = new CallCmd(Token.NoToken, logProcedure.Name, inParamsLog, new List<IdentifierExpr>());
      logAccessCallCmd.Proc = logProcedure;
      logAccessCallCmd.Attributes = SourceLocationAttributes.Clone() as QKeyValue;
      return logAccessCallCmd;
    }

    private void ExitWithNoSourceError(Variable v, AccessType Access)
    {
      Console.Error.WriteLine("No source location information available when processing " + 
        Access + " operation on " + v + " at " + GPUVerifyVCGenCommandLineOptions.inputFiles[0] + ":" + 
        v.tok.line + ":" + v.tok.col + ".  Aborting.");
      Environment.Exit(1);
    }

    private void MaybeAddValueParameter(List<Expr> parameters, AccessRecord ar, Expr Value, AccessType Access) {
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        if (Value != null) {
          parameters.Add(Value);
        }
        else {
          Expr e = Expr.Select(new IdentifierExpr(Token.NoToken, ar.v), new Expr[] { ar.Index });
          e.Type = (ar.v.TypedIdent.Type as MapType).Result;
          parameters.Add(e);
        }
      }
    }

    private void MaybeAddValueOldParameter(List<Expr> parameters, AccessRecord ar, AccessType Access) {
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
          Expr e = Expr.Select(new IdentifierExpr(Token.NoToken, ar.v), new Expr[] { ar.Index });
          e.Type = (ar.v.TypedIdent.Type as MapType).Result;
          parameters.Add(e);
      }
    }

    private CallCmd MakeUpdateBenignFlagCall(AccessRecord ar) {
      List<Expr> inParamsUpdateBenignFlag = new List<Expr>();
      inParamsUpdateBenignFlag.Add(ar.Index);
      Procedure updateBenignFlagProcedure = RI.GetRaceCheckingProcedure(Token.NoToken, "_UPDATE_WRITE_READ_BENIGN_FLAG_" + ar.v.Name);
      verifier.OnlyThread2.Add(updateBenignFlagProcedure.Name);
      CallCmd updateBenignFlagCallCmd = new CallCmd(Token.NoToken, updateBenignFlagProcedure.Name, inParamsUpdateBenignFlag, new List<IdentifierExpr>());
      updateBenignFlagCallCmd.Proc = updateBenignFlagProcedure;
      return updateBenignFlagCallCmd;
    }

  }


}
