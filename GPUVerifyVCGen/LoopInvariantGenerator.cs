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
   invariantGenerationRules.Add(new LoopVariableBoundsInvariantGenerator(verifier));
  }

  public static void PreInstrument(GPUVerifier verifier, Implementation impl)
  {
   foreach (var region in verifier.RootRegion(impl).SubRegions())
   {
    GenerateCandidateForReducedStrengthStrideVariables(verifier, impl, region);
    GenerateCandidateForNonNegativeGuardVariables(verifier, impl, region);
    GenerateCandidateForNonUniformGuardVariables(verifier, impl, region);
    GenerateCandidateForLoopBounds(verifier, impl, region);
   }
   GenerateCandidateForLoopsWhichAreControlDependentOnThreadOrGroupIDs(verifier, impl);
  }

  private static void GenerateCandidateForNonUniformGuardVariables(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   if (!verifier.ContainsBarrierCall(region))
    return;

   HashSet<Variable> partitionVars = region.PartitionVariablesOfHeader();
   HashSet<Variable> guardVars = new HashSet<Variable>();

   var formals = impl.InParams.Select(x => x.Name);
   var modset = GetModifiedVariables(region).Select(x => x.Name);
   foreach (var v in partitionVars)
   {
    Expr expr = verifier.varDefAnalyses[impl].DefOfVariableName(v.Name);
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
      x.TypedIdent.Type.Equals(Microsoft.Boogie.Type.GetBvType(verifier.size_t_bits))
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
       var inv = Expr.Eq(sub, new NAryExpr(Token.NoToken, new FunctionCall(otherbv32), args));
       verifier.AddCandidateInvariant(region, inv, "guardMinusInitialIsUniform", InferenceStages.BASIC_CANDIDATE_STAGE);
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
   foreach (var v in partitionVars)
   {
    var expr = verifier.varDefAnalyses[impl].DefOfVariableName(v.Name);
    if (!(expr is NAryExpr))
     continue;
    var nary = expr as NAryExpr;
    if (!(nary.Fun.FunctionName.Equals("BV32_SLE") ||
        nary.Fun.FunctionName.Equals("BV32_SLT") ||
        nary.Fun.FunctionName.Equals("BV32_SGE") ||
        nary.Fun.FunctionName.Equals("BV32_SGT")))
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
    int BVWidth = (v.TypedIdent.Type as BvType).Bits;
    // REVISIT: really we only want to guess for /integer/ variables.
    if (BVWidth >= 8)
    {
     var inv = verifier.IntRep.MakeSle(verifier.IntRep.GetLiteral(0, BVWidth), new IdentifierExpr(v.tok, v));
     verifier.AddCandidateInvariant(region, inv, "guardNonNeg", InferenceStages.BASIC_CANDIDATE_STAGE);
    }
   }
  }

  private static void GenerateCandidateForReducedStrengthStrideVariables(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   var rsa = verifier.reducedStrengthAnalyses[impl];
   foreach (string lc in rsa.StridedLoopCounters(region.Identifier()))
   {
    var sc = rsa.GetStrideConstraint(lc);
    Variable lcVariable = impl.LocVars.Where(Item => Item.Name == lc).ToList()[0];
    var lcExpr = new IdentifierExpr(Token.NoToken, lcVariable);
    var lcPred = sc.MaybeBuildPredicate(verifier, lcExpr);

    if (lcPred != null)
    {
     verifier.AddCandidateInvariant(region, lcPred, "loopCounterIsStrided", InferenceStages.BASIC_CANDIDATE_STAGE);
    }
   }
  }

  private static void GenerateCandidateForLoopBounds(GPUVerifier verifier, Implementation impl, IRegion region)
  {
   HashSet<Variable> modifiedVariables = GetModifiedVariables(region);
   // Get the partition variables associated with the header
   HashSet<Variable> partitionVars = region.PartitionVariablesOfHeader();
   foreach (Variable v in partitionVars)
   {
    // Find the expression which defines a particular partition variable.
    // Visit the expression and rip out any variable in the mod set of the loop.
    // We assume that any variable satisfying these conditions is a loop counter
    Expr partitionDefExpr = verifier.varDefAnalyses[impl].DefOfVariableName(v.Name);
    var visitor = new VariablesOccurringInExpressionVisitor();
    visitor.Visit(partitionDefExpr);
    HashSet<Variable> loopCounters = new HashSet<Variable>();
    foreach (Variable variable in visitor.GetVariables())
    {
        if (modifiedVariables.Contains(variable))
            loopCounters.Add(variable);
    }

    if (loopCounters.Count == 1)
    {
     Variable loopCounter = loopCounters.ToList()[0];
     foreach (Block preheader in region.PreHeaders())
     {
      foreach (AssignCmd cmd in preheader.Cmds.Where(x => x is AssignCmd).Reverse<Cmd>())
      {
       var lhss = cmd.Lhss.Where(x => x is SimpleAssignLhs);
       foreach (var LhsRhs in lhss.Zip(cmd.Rhss))
       {
        if (LhsRhs.Item1.DeepAssignedVariable.Name == loopCounter.Name)
        {
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSle(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound", InferenceStages.BASIC_CANDIDATE_STAGE);
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSge(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound", InferenceStages.BASIC_CANDIDATE_STAGE);
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUle(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound", InferenceStages.BASIC_CANDIDATE_STAGE);
         verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUge(new IdentifierExpr(loopCounter.tok, loopCounter), LhsRhs.Item2), "loopBound", InferenceStages.BASIC_CANDIDATE_STAGE);
        }
       }
      }
     }
    }
   }
  }

  private static void GenerateCandidateForLoopsWhichAreControlDependentOnThreadOrGroupIDs(GPUVerifier verifier, Implementation impl)
  {
   // We use control dependence information to determine whether a loop is always uniformly executed.
   // If a loop is control dependent on a conditional statement involving thread or group IDs then the following 
   // candidate invariants are produced:
   // 1) If the conditional expression causes control flow to avoid the loop, the thread is disabled
   // 2) If the conditional expression causes control flow to avoid the loop, the thread does not read or write 
   //    to any array within the loop
   // Note that, if the loop is control dependent on multiple nodes in the CFG, the LHS of these implications will be
   // a disjunction of the conditional expressions
   
   Graph<Block> cfg = Program.GraphFromImpl(impl);
   var ctrlDep = cfg.ControlDependence();
   ctrlDep.TransitiveClosure();
   Graph<Block> loopInfo = verifier.Program.ProcessLoops(impl);
   Dictionary<Block, AssignmentExpressionExpander> controlNodeExprInfo = new Dictionary<Block, AssignmentExpressionExpander>();
   
   foreach (Block header in loopInfo.Headers)
   {
    // Extract the body associated with this loop
    HashSet<Block> loopBody = new HashSet<Block>();
    foreach (Block tail in loopInfo.BackEdgeNodes(header))
     loopBody.UnionWith(loopInfo.NaturalLoops(header, tail));
   
    // Go through every loop in the CFG and determine the distinct basic blocks on which it is control dependent
    HashSet<Block> controllingBlocks = new HashSet<Block>();
    foreach (Block block in cfg.Nodes)
    {
     if (ctrlDep.ContainsKey(block) && block != header)
     {
      // Loop headers are always control dependent on themselves. Ignore this control dependence
      // as we want to compute the set of basic blocks whose conditions lead to execution of the loop header
      // at least once
      foreach (Block ctrlDepBlock in ctrlDep[block].Where(x => x == header))
      {
       controllingBlocks.Add(block);
      }
     }
    }
    if (controllingBlocks.Count > 0)
    {
     // If the loop is control dependent on other blocks
     HashSet<Block> toKeep = new HashSet<Block>();
     foreach (Block controlling in controllingBlocks)
     {
      if (!controlNodeExprInfo.ContainsKey(controlling))
      {
       // If we have not done so already, fully expand the expression which determines the direction of the conditional
       HashSet<Variable> tempVariables = new HashSet<Variable>();
       var visitor = new VariablesOccurringInExpressionVisitor();
       foreach (Block succ in cfg.Successors(controlling))
       {
        foreach (var assume in succ.Cmds.OfType<AssumeCmd>().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "partition")))
        {
         visitor.Visit(assume.Expr);
         tempVariables.UnionWith(visitor.GetVariables().ToList());
        }
       }
       
       if (tempVariables.Count > 0)
       {
        toKeep.Add(controlling);
        // There should be exactly one partition variable 
        Debug.Assert(tempVariables.Count == 1);
        controlNodeExprInfo[controlling] = new AssignmentExpressionExpander(cfg, tempVariables.Single());
       }
      }
     }
     
     controllingBlocks.IntersectWith(toKeep);
     
     // Build up the expressions on the LHS of the implication
     List<Expr> antecedentExprs = new List<Expr>();
     foreach (Block controlling in controllingBlocks)
     {
      if (controlNodeExprInfo[controlling].GetGPUVariables().Count > 0)
      {
       // If the fully-expanded conditional expression contains thread or group IDs
       foreach (Block succ in cfg.Successors(controlling))
       {
         // Find the side of the branch that must execute if the loop executes
        if (cfg.DominatorMap.DominatedBy(header, succ))
        {
         // Get the assume associated with the branch
         var assume = succ.Cmds.OfType<AssumeCmd>().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "partition")).Single();
         if (assume.Expr is NAryExpr)
         {
          NAryExpr nary = assume.Expr as NAryExpr;
          Debug.Assert((nary.Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not);
          // The assume is a not expression !E. Negating !E gives !!E which is logically equivalent to E.
          // E is the expression that must hold for the conditional to bypass the loop
          antecedentExprs.Add(controlNodeExprInfo[controlling].GetUnexpandedExpr());
         }
         else
         {
          Expr negatedExpr = Expr.Not(controlNodeExprInfo[controlling].GetUnexpandedExpr());
          antecedentExprs.Add(negatedExpr);
         }         
        }
       }
      }
     }
     if (antecedentExprs.Count > 0)
     {
      // Create the set of implications which state that, if one of the antecedent expressions holds, 
      // the thread is not enabled and does not read or write to an array location in the loop body
      Expr lhsOfImplication;
      if (antecedentExprs.Count > 1)
      {
       lhsOfImplication = Expr.Or(antecedentExprs[0], antecedentExprs[1]);
       for (int i = 2; i < antecedentExprs.Count; ++i)
       {
        lhsOfImplication = Expr.Or(lhsOfImplication, antecedentExprs[i]);
       }
      }
      else
       lhsOfImplication = antecedentExprs[0];
      AddInvariantsForLoopsWhichAreControlDependentOnThreadOrGroupIDs(verifier, header, loopBody, lhsOfImplication);
     }
    }
   }
  }

  private static void AddInvariantsForLoopsWhichAreControlDependentOnThreadOrGroupIDs(GPUVerifier verifier, Block header, HashSet<Block> loopBody, Expr lhsOfImplication)
  {
   // Invariant #1: The thread is not enabled 
   string enabledVariableName = "__enabled";
   Variable enabledVariable = (Variable)verifier.ResContext.LookUpVariable(enabledVariableName);
   if (enabledVariable == null)
   {
    enabledVariable = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, enabledVariableName, Microsoft.Boogie.Type.Bool), false);
    enabledVariable.AddAttribute("__enabled");
    verifier.ResContext.AddVariable(enabledVariable, true);
   }
   Expr invariantEnabled = Expr.Imp(lhsOfImplication, Expr.Not(new IdentifierExpr(Token.NoToken, enabledVariable)));
   PredicateCmd enabledPredicate = verifier.Program.CreateCandidateInvariant(invariantEnabled, "conditionalLoopExecution", InferenceStages.BASIC_CANDIDATE_STAGE);
   enabledPredicate.Attributes = new QKeyValue(Token.NoToken,"do_not_predicate", new List<object>() { }, enabledPredicate.Attributes);
   header.Cmds.Insert(0, enabledPredicate);
   
   // Retrieve the variables read and written in the loop body
   var readVisitor = new VariablesOccurringInExpressionVisitor();
   var writeVisitor = new VariablesOccurringInExpressionVisitor();
   foreach (Block bodyBlock in loopBody)
   {
    foreach (AssignCmd assignment in bodyBlock.Cmds.OfType<AssignCmd>())
    {
     var mapLhss = assignment.Lhss.OfType<MapAssignLhs>();
     foreach (var LhsRhs in mapLhss.Zip(assignment.Rhss))
     {
      writeVisitor.Visit(LhsRhs.Item1);
      readVisitor.Visit(LhsRhs.Item2);
     }
     var simpleLhss = assignment.Lhss.OfType<SimpleAssignLhs>();
     foreach (var LhsRhs in simpleLhss.Zip(assignment.Rhss))
     {
      readVisitor.Visit(LhsRhs.Item2);
     }
    }
   }
     
   // Invariant #2: Arrays in global or group-shared memory are not written
   foreach (Variable found in writeVisitor.GetVariables().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "global") 
      || QKeyValue.FindBoolAttribute(x.Attributes, "group_shared")))
   {
    Variable writeHasOccurredVariable = (Variable)verifier.ResContext.LookUpVariable("_WRITE_HAS_OCCURRED_" + found.Name);
    Debug.Assert(writeHasOccurredVariable != null);
    Expr writeHasNotOccurred = Expr.Imp(lhsOfImplication, Expr.Not(new IdentifierExpr(Token.NoToken, writeHasOccurredVariable)));
    PredicateCmd writePredicate = verifier.Program.CreateCandidateInvariant(writeHasNotOccurred, "conditionalLoopExecution", InferenceStages.BASIC_CANDIDATE_STAGE);
    writePredicate.Attributes = new QKeyValue(Token.NoToken,"do_not_predicate", new List<object>() { }, writePredicate.Attributes);
    header.Cmds.Insert(0, writePredicate);
   }
     
   // Invariant #3: Arrays in global or group-shared memory are not read
   foreach (Variable found in readVisitor.GetVariables().Where(x => QKeyValue.FindBoolAttribute(x.Attributes, "global") 
      || QKeyValue.FindBoolAttribute(x.Attributes, "group_shared")))
   {
    Variable readHasOccurredVariable = (Variable)verifier.ResContext.LookUpVariable("_READ_HAS_OCCURRED_" + found.Name);
    Debug.Assert(readHasOccurredVariable != null);
    Expr readHasNotOccurred = Expr.Imp(lhsOfImplication, Expr.Not(new IdentifierExpr(Token.NoToken, readHasOccurredVariable)));
    PredicateCmd readPredicate = verifier.Program.CreateCandidateInvariant(readHasNotOccurred, "conditionalLoopExecution", InferenceStages.BASIC_CANDIDATE_STAGE);
    readPredicate.Attributes = new QKeyValue(Token.NoToken,"do_not_predicate", new List<object>() { }, readPredicate.Attributes);
    header.Cmds.Insert(0, readPredicate);
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

   AddCandidateInvariants(verifier.RootRegion(Impl), LocalVars, Impl);
  }

  private void AddEqualityCandidateInvariant(IRegion region, string LoopPredicate, Variable v)
  {
   verifier.AddCandidateInvariant(region,
    Expr.Eq(
     new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable)),
     new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable))
    ), "equality", InferenceStages.BASIC_CANDIDATE_STAGE);
  }

  private void AddPredicatedEqualityCandidateInvariant(IRegion region, string LoopPredicate, Variable v)
  {
   verifier.AddCandidateInvariant(region, Expr.Imp(
    Expr.And(
     new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$1", Microsoft.Boogie.Type.Int))),
     new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, LoopPredicate + "$2", Microsoft.Boogie.Type.Int)))
    ),
    Expr.Eq(
     new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable)),
     new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier.uniformityAnalyser, Impl.Name).VisitVariable(v.Clone() as Variable))
    )), "predicatedEquality", InferenceStages.BASIC_CANDIDATE_STAGE);
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

    verifier.AddCandidateInvariant(region, uniformEnabledPredicate, "loopPredicateEquality", InferenceStages.BASIC_CANDIDATE_STAGE);

    verifier.AddCandidateInvariant(region, Expr.Imp(GPUVerifier.ThreadsInSameGroup(), uniformEnabledPredicate), "loopPredicateEquality", InferenceStages.BASIC_CANDIDATE_STAGE);

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

  private void AddCandidateInvariants(IRegion region, HashSet<Variable> LocalVars, Implementation Impl)
  {
   foreach (IRegion subregion in region.SubRegions())
   {
    foreach (InvariantGenerationRule r in invariantGenerationRules)
    {
     r.GenerateCandidates(Impl, subregion);
    }

    AddBarrierDivergenceCandidates(LocalVars, Impl, subregion);

    verifier.RaceInstrumenter.AddRaceCheckingCandidateInvariants(Impl, subregion);

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
 }
}
