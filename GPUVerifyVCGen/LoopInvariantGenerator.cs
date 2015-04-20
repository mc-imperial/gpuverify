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
using Microsoft.Boogie.GraphUtil;
using System.Diagnostics;
using System.Text.RegularExpressions;
using GPUVerify.InvariantGenerationRules;

namespace GPUVerify
{
 class LoopInvariantGenerator
 {
  private GPUVerifier verifier;
  private Implementation Impl;
  private List<InvariantGenerationRule> invariantGenerationRules;

  LoopInvariantGenerator(GPUVerifier verifier, Implementation Impl)
  {
   this.verifier = verifier;
   this.Impl = Impl;

   invariantGenerationRules = new List<InvariantGenerationRule>();
   invariantGenerationRules.Add(new PowerOfTwoInvariantGenerator(verifier));
  }

  public static void EstablishDisabledLoops(GPUVerifier verifier, Implementation impl)
  {
   foreach (var region in verifier.RootRegion(impl).SubRegions())
   {
    if (!AccessesGlobalArrayOrUnsafeBarrier(region, verifier)) {
     verifier.AddRegionWithLoopInvariantsDisabled(region);
    }
   }
  }

  public static void PreInstrument(GPUVerifier verifier, Implementation impl)
  {
   foreach (var region in verifier.RootRegion(impl).SubRegions())
   {
    if (verifier.RegionHasLoopInvariantsDisabled(region))
     continue;

    GenerateCandidateForReducedStrengthStrideVariables(verifier, impl, region);
    GenerateCandidateForNonNegativeGuardVariables(verifier, impl, region);
    GenerateCandidateForNonUniformGuardVariables(verifier, impl, region);
    GenerateCandidateForLoopBounds(verifier, impl, region);
    GenerateCandidateForEnabledness(verifier, impl, region);
    GenerateCandidateForEnablednessWhenAccessingSharedArrays(verifier, impl, region);
   }
  }

  private static void GenerateCandidateForEnablednessWhenAccessingSharedArrays(GPUVerifier verifier, Implementation impl, IRegion region) {
    Block header = region.Header();
    if(verifier.uniformityAnalyser.IsUniform(impl.Name, header)) {
      return;
    }

    var CFG = Program.GraphFromImpl(impl);
    Dictionary<Block, HashSet<Block>> ControlDependence = CFG.ControlDependence();
    ControlDependence.TransitiveClosure();
    CFG.ComputeLoops();

    List<Expr> Guards = new List<Expr>();
    foreach(var b in ControlDependence.Keys.Where(Item => ControlDependence[Item].Contains(region.Header()))) {
      foreach(var succ in CFG.Successors(b).Where(Item => CFG.DominatorMap.DominatedBy(header, Item))) {
        var Guard = MaybeExtractGuard(verifier, impl, succ);
        if(Guard != null) {
          Guards.Add(Guard);
          break;
        }
      }
    }

    if(Guards.Count == 0) {
      return;
    }

    IEnumerable<Variable> ReadVariables;
    IEnumerable<Variable> WrittenVariables;
    GetReadAndWrittenVariables(region, out ReadVariables, out WrittenVariables);

    foreach(var v in ReadVariables.Where(Item => verifier.KernelArrayInfo.getAllNonLocalArrays().Contains(Item)
      && !verifier.KernelArrayInfo.getReadOnlyNonLocalArrays().Contains(Item))) {
      foreach(var g in Guards) {
        verifier.AddCandidateInvariant(region,
          Expr.Imp(Expr.Ident(verifier.FindOrCreateAccessHasOccurredVariable(v.Name, AccessType.READ)),
                    g), "accessOnlyIfEnabledInEnclosingScopes", "do_not_predicate");
      }
    }

    foreach(var v in WrittenVariables.Where(Item => verifier.KernelArrayInfo.getAllNonLocalArrays().Contains(Item))) {
      foreach(var g in Guards) {
        verifier.AddCandidateInvariant(region,
          Expr.Imp(Expr.Ident(verifier.FindOrCreateAccessHasOccurredVariable(v.Name, AccessType.WRITE)),
                    g), "accessOnlyIfEnabledInEnclosingScopes", "do_not_predicate");
      }
    }
  }

  private static void GetReadAndWrittenVariables(IRegion region, out IEnumerable<Variable> ReadVariables, out IEnumerable<Variable> WrittenVariables) {
    var readVisitor = new VariablesOccurringInExpressionVisitor();
    var writeVisitor = new VariablesOccurringInExpressionVisitor();
    foreach (AssignCmd assignment in region.Cmds().OfType<AssignCmd>()) {
      var mapLhss = assignment.Lhss.OfType<MapAssignLhs>();
      foreach (var LhsRhs in mapLhss.Zip(assignment.Rhss)) {
        writeVisitor.Visit(LhsRhs.Item1);
        readVisitor.Visit(LhsRhs.Item2);
      }
      var simpleLhss = assignment.Lhss.OfType<SimpleAssignLhs>();
      foreach (var LhsRhs in simpleLhss.Zip(assignment.Rhss)) {
        readVisitor.Visit(LhsRhs.Item2);
      }
    }
    ReadVariables = readVisitor.GetVariables();
    WrittenVariables = writeVisitor.GetVariables();
  }

  private static void GenerateCandidateForEnabledness(GPUVerifier verifier, Implementation impl, IRegion region) {
    Block header = region.Header();
    if(verifier.uniformityAnalyser.IsUniform(impl.Name, header)) {
      return;
    }

    var CFG = Program.GraphFromImpl(impl);
    Dictionary<Block, HashSet<Block>> ControlDependence = CFG.ControlDependence();
    ControlDependence.TransitiveClosure();
    CFG.ComputeLoops();
    var LoopNodes = CFG.BackEdgeNodes(header).Select(Item => CFG.NaturalLoops(header, Item)).SelectMany(Item => Item);

    Expr GuardEnclosingLoop = null;
    foreach(var b in ControlDependence.Keys.Where(Item => ControlDependence[Item].Contains(region.Header()))) {
      foreach(var succ in CFG.Successors(b).Where(Item => CFG.DominatorMap.DominatedBy(header, Item))) {
        var Guard = MaybeExtractGuard(verifier, impl, succ);
        if(Guard != null) {
          GuardEnclosingLoop = GuardEnclosingLoop == null ? Guard : Expr.And(GuardEnclosingLoop, Guard);
          break;
        }
      }
    }

    if(GuardEnclosingLoop != null) {
      verifier.AddCandidateInvariant(region, Expr.Imp(Expr.Ident(verifier.FindOrCreateEnabledVariable()), GuardEnclosingLoop), "conditionsImpliedByEnabledness");
    }

    var DualCFG = CFG.Dual(new Block());
    Block LoopConditionDominator = header;

    // The dominator might have multiple successors
    while (CFG.Successors(LoopConditionDominator).Count(Item => LoopNodes.Contains(Item)) > 1) {
      // Find the immediate post-dominator of the successors
      Block block = null;
      foreach(var succ in CFG.Successors(LoopConditionDominator).Where(Item => LoopNodes.Contains(Item))) {
        if (block == null)
          block = succ;
        else
          block = DualCFG.DominatorMap.LeastCommonAncestor(block, succ);
      }
      // Use the immediate post-dominator
      LoopConditionDominator = block;
    }

    Expr GuardIncludingLoopCondition = null;
    foreach(var succ in CFG.Successors(LoopConditionDominator).Where(Item => LoopNodes.Contains(Item))) {
      var Guard = MaybeExtractGuard(verifier, impl, succ);
      if(Guard != null) {
        // There is at most one successor, so it's safe not use use GuardIncludingLoopCondition ont the rhs
        GuardIncludingLoopCondition = GuardEnclosingLoop == null ? Guard : Expr.And(GuardEnclosingLoop, Guard);
        break;
      }
    }

    if(GuardIncludingLoopCondition != null) {
      verifier.AddCandidateInvariant(region, Expr.Imp(GuardIncludingLoopCondition, Expr.Ident(verifier.FindOrCreateEnabledVariable())), "conditionsImplyingEnabledness", "do_not_predicate");
    }

  }


  private static Expr MaybeExtractGuard(GPUVerifier verifier, Implementation impl, Block b) {
    if (b.Cmds.Count() > 0) {
      var a = b.Cmds[0] as AssumeCmd;
      if (a != null && QKeyValue.FindBoolAttribute(a.Attributes, "partition")) {
        if (a.Expr is IdentifierExpr) {
          return verifier.varDefAnalysesRegion[impl].DefOfVariableName(((IdentifierExpr)a.Expr).Name);
        } else if(a.Expr is NAryExpr) {
          var nary = (NAryExpr)a.Expr;
          if (nary.Fun is UnaryOperator &&
              (nary.Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not &&
              nary.Args[0] is IdentifierExpr) {
            var d = verifier.varDefAnalysesRegion[impl].DefOfVariableName(((IdentifierExpr)(a.Expr as NAryExpr).Args[0]).Name);
            if (d == null)
              return null;
            else
              return Expr.Not(d);
          }
        }
      }
    }
    return null;
  }

  private static void GenerateCandidateForNonUniformGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   if (!verifier.ContainsBarrierCall(region) && !GPUVerifyVCGenCommandLineOptions.WarpSync)
    return;

   HashSet<Variable> partitionVars = region.PartitionVariablesOfHeader();
   HashSet<Variable> guardVars = new HashSet<Variable>();

   var formals = impl.InParams.Select(x => x.Name);
   var modset = GetModifiedVariables(region).Select(x => x.Name);
   foreach (var v in partitionVars)
   {
    Expr expr = verifier.varDefAnalysesRegion[impl].DefOfVariableName(v.Name);
    if (expr == null)
     continue;
    var visitor = new VariablesOccurringInExpressionVisitor();
    visitor.Visit(expr);
    guardVars.UnionWith(
     visitor.GetVariables().Where(
      x => x.Name.StartsWith("$") &&
      !formals.Contains(x.Name) &&
      modset.Contains(x.Name) &&
      !verifier.uniformityAnalyser.IsUniform(impl.Name, x.Name) &&
      x.TypedIdent.Type.IsBv &&
      (x.TypedIdent.Type.BvBits % 8 == 0)
     )
    );
   }
   List<AssignCmd> assignments = new List<AssignCmd>();
   foreach (Block b in region.PreHeaders())
   {
    foreach (AssignCmd c in b.Cmds.Where(x => x is AssignCmd))
    {
     assignments.Add(c);
    }
   }
   foreach (var v in guardVars)
   {
    foreach (AssignCmd c in assignments)
    {
     foreach (var a in c.Lhss.Zip(c.Rhss))
     {
      var lhs = a.Item1;
      var rhs = a.Item2;
      if (!(lhs is SimpleAssignLhs))
       continue;
      var sLhs = (SimpleAssignLhs)lhs;
      var theVar = sLhs.DeepAssignedVariable;
      if (theVar.Name == v.Name)
      {
       var sub = verifier.IntRep.MakeSub(new IdentifierExpr(Token.NoToken, v), rhs as Expr);
       List<Expr> args = new List<Expr>();
       args.Add(sub);
       Function otherbv = verifier.FindOrCreateOther(sub.Type.BvBits);
       var inv = Expr.Eq(sub, new NAryExpr(Token.NoToken, new FunctionCall(otherbv), args));
       verifier.AddCandidateInvariant(region, inv, "guardMinusInitialIsUniform");
       var groupInv = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), inv);
       verifier.AddCandidateInvariant(region, groupInv, "guardMinusInitialIsUniform");
      }
     }
    }
   }
  }

  private static void GenerateCandidateForNonNegativeGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   HashSet<Variable> partitionVars = region.PartitionVariablesOfHeader();
   HashSet<Variable> nonnegVars = new HashSet<Variable>();

   var formals = impl.InParams.Select(x => x.Name);
   var modset = GetModifiedVariables(region).Select(x => x.Name);
   Regex pattern = new Regex(@"\bBV\d*_((SLE)|(SLT)|(SGE)|(SGT))\b");
   foreach (var v in partitionVars)
   {
    var expr = verifier.varDefAnalysesRegion[impl].DefOfVariableName(v.Name);
    if (!(expr is NAryExpr))
     continue;
    var nary = expr as NAryExpr;
    if (!pattern.Match(nary.Fun.FunctionName).Success)
     continue;
    var visitor = new VariablesOccurringInExpressionVisitor();
    visitor.Visit(nary);
    nonnegVars.UnionWith(
     visitor.GetVariables().Where(
      x => x.Name.StartsWith("$") &&
      !formals.Contains(x.Name) &&
      modset.Contains(x.Name) &&
      x.TypedIdent.Type.IsBv
     )
    );
   }
   foreach (var v in nonnegVars)
   {
    int BVWidth = v.TypedIdent.Type.BvBits;
    // REVISIT: really we only want to guess for /integer/ variables.
    if (BVWidth >= 8)
    {
     var inv = verifier.IntRep.MakeSle(verifier.Zero(BVWidth), new IdentifierExpr(v.tok, v));
     verifier.AddCandidateInvariant(region, inv, "guardNonNeg");
    }
   }
  }

  private static void GenerateCandidateForReducedStrengthStrideVariables(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   var rsa = verifier.reducedStrengthAnalysesRegion[impl];
   var regionId = region.Identifier();
   foreach (string iv in rsa.StridedInductionVariables(regionId))
   {
    var sc = rsa.GetStrideConstraint(iv, regionId);
    Variable ivVariable = impl.LocVars.Where(Item => Item.Name == iv).ToList()[0];
    var ivExpr = new IdentifierExpr(Token.NoToken, ivVariable);
    var ivPred = sc.MaybeBuildPredicate(verifier, ivExpr);

    if (ivPred != null)
    {
     verifier.AddCandidateInvariant(region, ivPred, "loopCounterIsStrided");
    }
   }
  }

  private static void GenerateCandidateForLoopBounds(GPUVerifier verifier, Implementation impl, IRegion region)
    {
      HashSet<Variable> loopCounters = new HashSet<Variable>();
      HashSet<Variable> modifiedVariables = GetModifiedVariables(region);
      // Get the partition variables associated with the header
      HashSet<Variable> partitionVars = region.PartitionVariablesOfRegion();
      foreach (Variable v in partitionVars)
      {
        // Find the expression which defines a particular partition variable.
        // Visit the expression and select any variable in the mod set of the loop.
        // We assume that any variable satisfying these conditions is a loop counter
        Expr partitionDefExpr = verifier.varDefAnalysesRegion[impl].DefOfVariableName(v.Name);
        if (partitionDefExpr == null) // multiple definitions or no definition
            continue;
        var visitor = new VariablesOccurringInExpressionVisitor();
        visitor.Visit(partitionDefExpr);
        foreach (Variable variable in visitor.GetVariables())
        {
          if (modifiedVariables.Contains(variable))
          {
            loopCounters.Add(variable);
          }
        }
      }

    foreach (Variable loopCounter in loopCounters)
    {
     foreach (Block preheader in region.PreHeaders())
     {
      foreach (AssignCmd cmd in preheader.Cmds.Where(x => x is AssignCmd).Reverse<Cmd>())
      {
       var lhss = cmd.Lhss.Where(x => x is SimpleAssignLhs);
       foreach (var LhsRhs in lhss.Zip(cmd.Rhss))
       {
        if (LhsRhs.Item1.DeepAssignedVariable.Name == loopCounter.Name)
        {
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSle(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound");
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSge(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound");
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUle(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound");
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUge(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound");
        }
       }
      }
     }
   }
  }

  public static void PostInstrument(GPUVerifier verifier, Implementation Impl)
  {
   new LoopInvariantGenerator(verifier, Impl).PostInstrument();
  }

  internal void PostInstrument()
  {
   HashSet<Variable> LocalVars = new HashSet<Variable>();
   foreach (Variable v in Impl.LocVars)
   {
    LocalVars.Add(v);
   }
   foreach (Variable v in Impl.InParams)
   {
    LocalVars.Add(v);
   }
   foreach (Variable v in Impl.OutParams)
   {
    LocalVars.Add(v);
   }

   AddCandidateInvariants(LocalVars, Impl);
  }

  private void AddPredicatedEqualityCandidateInvariant(IRegion region, string LoopPredicate, Variable v)
  {
   var inv = Expr.Imp(
    Expr.And(
     new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$1", Microsoft.Boogie.Type.Int))),
     new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$2", Microsoft.Boogie.Type.Int)))
    ),
    Expr.Eq(
     new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable)),
     new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable))
    ));

   verifier.AddCandidateInvariant(region, inv, "predicatedEquality");
  }

  private Dictionary<string, int> GetAssignmentCounts(Implementation impl)
  {

   Dictionary<string, int> result = new Dictionary<string, int>();

   foreach (var c in verifier.RootRegion(impl).Cmds())
   {
    if (c is AssignCmd)
    {
     var aCmd = (AssignCmd)c;
     HashSet<string> alreadySeenInThisAssignment = new HashSet<string>();
     foreach (var a in aCmd.Lhss)
     {
      if (a is SimpleAssignLhs)
      {
       var v = GVUtil.StripThreadIdentifier(
                ((SimpleAssignLhs)a).AssignedVariable.Name);
       if (!alreadySeenInThisAssignment.Contains(v))
       {
        if (result.ContainsKey(v))
        {
         result[v]++;
        }
        else
        {
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

   if (!verifier.ContainsBarrierCall(region) && !GPUVerifyVCGenCommandLineOptions.WarpSync)
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

    verifier.AddCandidateInvariant(region, uniformEnabledPredicate, "loopPredicateEquality");

    verifier.AddCandidateInvariant(region, Expr.Imp(GPUVerifier.ThreadsInSameGroup(), uniformEnabledPredicate), "loopPredicateEquality");

    Dictionary<string, int> assignmentCounts = GetAssignmentCounts(Impl);

    HashSet<string> alreadyConsidered = new HashSet<String>();

    foreach (var v in LocalVars)
    {
     string lv = GVUtil.StripThreadIdentifier(v.Name);
     if (alreadyConsidered.Contains(lv))
     {
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

     if (!assignmentCounts.ContainsKey(lv) || assignmentCounts[lv] <= 1)
     {
      continue;
     }

     if (!verifier.ContainsNamedVariable(
          GetModifiedVariables(region), lv))
     {
      continue;
     }

     AddPredicatedEqualityCandidateInvariant(region, LoopPredicate, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, lv, Microsoft.Boogie.Type.Int)));
    }

   }
  }

  private static bool IsDisjunctionOfPredicates(Expr guard)
  {
   if (!(guard is NAryExpr))
   {
    return false;
   }
   NAryExpr nary = (NAryExpr)guard;
   if (nary.Args.Count() != 2)
   {
    return false;
   }
   if (!(nary.Fun is BinaryOperator))
   {
    return false;
   }
   BinaryOperator binOp = (BinaryOperator)nary.Fun;
   if (binOp.Op != BinaryOperator.Opcode.Or)
   {
    return false;
   }
   if (!(nary.Args[0] is IdentifierExpr && nary.Args[1] is IdentifierExpr))
   {
    return false;
   }
   return GPUVerifier.IsPredicate(GVUtil.StripThreadIdentifier(
    ((IdentifierExpr)nary.Args[0]).Name)) &&
   GPUVerifier.IsPredicate(GVUtil.StripThreadIdentifier(
    ((IdentifierExpr)nary.Args[1]).Name));
  }

  private void AddCandidateInvariants(HashSet<Variable> LocalVars, Implementation Impl)
  {
   foreach (IRegion region in verifier.RootRegion(Impl).SubRegions())
   {
    if (verifier.RegionHasLoopInvariantsDisabled(region))
     continue;

    foreach (InvariantGenerationRule r in invariantGenerationRules)
     r.GenerateCandidates(Impl, region);

    AddBarrierDivergenceCandidates(LocalVars, Impl, region);

    verifier.RaceInstrumenter.AddRaceCheckingCandidateInvariants(Impl, region);
   }
  }

  internal static HashSet<Variable> GetModifiedVariables(IRegion region)
  {
   HashSet<Variable> result = new HashSet<Variable>();

   foreach (Cmd c in region.Cmds())
   {
    List<Variable> vars = new List<Variable>();
    c.AddAssignedVariables(vars);
    foreach (Variable v in vars)
    {
     Debug.Assert(v != null);
     result.Add(v);
    }
   }

   return result;
  }

  internal static bool AccessesGlobalArrayOrUnsafeBarrier(Cmd c, GPUVerifier verifier)
  {
   var StateToCheck = verifier.KernelArrayInfo;

   if (c is CallCmd)
   {
    // Speculate invariants if we see atomics, async_work_group_copy, and
    // wait_group_events, which relate to race checking
    CallCmd call = c as CallCmd;
    if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic"))
     return true;

    if (QKeyValue.FindBoolAttribute(call.Attributes, "async_work_group_copy"))
     return true;

    if (QKeyValue.FindBoolAttribute(call.Attributes, "wait_group_events"))
     return true;

    // Speculate invariants if we see an unsafe barrier,
    // which we need to check for barrier divergence
    if (GPUVerifier.IsBarrier(call.Proc) &&
        !QKeyValue.FindBoolAttribute(call.Proc.Attributes, "safe_barrier"))
     return true;

    // Speculate invariants if we see a call to a procedure that has a non-local array
    // or constant array in its modset
    List<Variable> vars =  new List<Variable>();
    call.AddAssignedVariables(vars);
    foreach (Variable v in vars)
    {
     if (StateToCheck.getAllNonLocalArrays().Contains(v))
      return true;
     if (StateToCheck.getConstantArrays().Contains(v))
      return true;
    }
   }

   // Speculate invariants if race instrumentation or a constant write
   // instrumentation will occur
   if (c is AssignCmd)
   {
    AssignCmd assign = c as AssignCmd;

    ReadCollector rc = new ReadCollector(StateToCheck);
    foreach (var rhs in assign.Rhss)
     rc.Visit(rhs);
    foreach (var access in rc.nonPrivateAccesses)
    {
     if (!StateToCheck.getReadOnlyNonLocalArrays().Contains(access.v))
      return true;
    }

    foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss))
    {
     WriteCollector wc = new WriteCollector(StateToCheck);
     wc.Visit(LhsRhs.Item1);
     if (wc.FoundNonPrivateWrite())
      return true;
    }

    foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss))
    {
     ConstantWriteCollector cwc = new ConstantWriteCollector(StateToCheck);
     cwc.Visit(LhsRhs.Item1);
     if (cwc.FoundWrite())
      return true;
    }
   }

   // Speculate invariants if we see an assert that is not a sourceloc or
   // block_sourceloc assert; such asserts is likely user supplied.
   if (c is AssertCmd)
   {
    AssertCmd assertion = c as AssertCmd;
    if (!QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc") &&
        !QKeyValue.FindBoolAttribute(assertion.Attributes, "block_sourceloc"))
     return true;
   }

   // Speculate invariants if we see an assume that is not a partition; such
   // an assume is likely user supplied.
   if (c is AssumeCmd)
   {
    AssumeCmd assumption = c as AssumeCmd;
    if (!QKeyValue.FindBoolAttribute(assumption.Attributes, "partition"))
     return true;
   }

   return false;
  }

  internal static bool AccessesGlobalArrayOrUnsafeBarrier(IRegion region, GPUVerifier verifier)
  {
   // Heuristic to establish whether to speculate loop invariants for a specific loop
   // based on the commands that occur int the loop.
   foreach (Cmd c in region.Cmds())
   {
    if (AccessesGlobalArrayOrUnsafeBarrier(c, verifier))
     return true;
   }
   return false;
  }
 }
}
