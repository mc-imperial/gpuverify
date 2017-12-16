//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using System.Collections.Generic;
    using System.Linq;
    using System.Text.RegularExpressions;
    using InvariantGenerationRules;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;

    public class LoopInvariantGenerator
    {
        private GPUVerifier verifier;
        private Implementation impl;
        private List<InvariantGenerationRule> invariantGenerationRules;

        private LoopInvariantGenerator(GPUVerifier verifier, Implementation impl)
        {
            this.verifier = verifier;
            this.impl = impl;

            invariantGenerationRules = new List<InvariantGenerationRule>();
            invariantGenerationRules.Add(new PowerOfTwoInvariantGenerator(verifier));
        }

        public static void EstablishDisabledLoops(GPUVerifier verifier, Implementation impl)
        {
            foreach (var region in verifier.RootRegion(impl).SubRegions())
            {
                if (!AccessesGlobalArrayOrUnsafeBarrier(region, verifier))
                    verifier.AddRegionWithLoopInvariantsDisabled(region);
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

        private static void GenerateCandidateForEnablednessWhenAccessingSharedArrays(GPUVerifier verifier, Implementation impl, IRegion region)
        {
            Block header = region.Header();
            if (verifier.UniformityAnalyser.IsUniform(impl.Name, header))
                return;

            var cfg = Program.GraphFromImpl(impl);
            Dictionary<Block, HashSet<Block>> controlDependence = cfg.ControlDependence();
            controlDependence.TransitiveClosure();
            cfg.ComputeLoops();

            List<Expr> guards = new List<Expr>();
            foreach (var b in controlDependence.Keys.Where(item => controlDependence[item].Contains(region.Header())))
            {
                foreach (var succ in cfg.Successors(b).Where(item => cfg.DominatorMap.DominatedBy(header, item)))
                {
                    var guard = MaybeExtractGuard(verifier, impl, succ);
                    if (guard != null)
                    {
                        guards.Add(guard);
                        break;
                    }
                }
            }

            if (guards.Count == 0)
            {
                return;
            }

            IEnumerable<Variable> readVariables;
            IEnumerable<Variable> writtenVariables;
            GetReadAndWrittenVariables(region, out readVariables, out writtenVariables);

            foreach (var v in readVariables.Where(item => verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(item)
                && !verifier.KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(item)))
            {
                foreach (var g in guards)
                {
                    verifier.AddCandidateInvariant(
                        region,
                        Expr.Imp(Expr.Ident(verifier.FindOrCreateAccessHasOccurredVariable(v.Name, AccessType.READ)), g),
                        "accessOnlyIfEnabledInEnclosingScopes",
                        "do_not_predicate");
                }
            }

            foreach (var v in writtenVariables.Where(item => verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(item)))
            {
                foreach (var g in guards)
                {
                    verifier.AddCandidateInvariant(
                        region,
                        Expr.Imp(Expr.Ident(verifier.FindOrCreateAccessHasOccurredVariable(v.Name, AccessType.WRITE)), g),
                        "accessOnlyIfEnabledInEnclosingScopes",
                        "do_not_predicate");
                }
            }
        }

        private static void GetReadAndWrittenVariables(IRegion region, out IEnumerable<Variable> readVariables, out IEnumerable<Variable> writtenVariables)
        {
            var readVisitor = new VariablesOccurringInExpressionVisitor();
            var writeVisitor = new VariablesOccurringInExpressionVisitor();
            foreach (AssignCmd assignment in region.Cmds().OfType<AssignCmd>())
            {
                var mapLhss = assignment.Lhss.OfType<MapAssignLhs>();
                foreach (var lhsRhs in mapLhss.Zip(assignment.Rhss))
                {
                    writeVisitor.Visit(lhsRhs.Item1);
                    readVisitor.Visit(lhsRhs.Item2);
                }

                var simpleLhss = assignment.Lhss.OfType<SimpleAssignLhs>();
                foreach (var lhsRhs in simpleLhss.Zip(assignment.Rhss))
                {
                    readVisitor.Visit(lhsRhs.Item2);
                }
            }

            readVariables = readVisitor.GetVariables();
            writtenVariables = writeVisitor.GetVariables();
        }

        private static void GenerateCandidateForEnabledness(GPUVerifier verifier, Implementation impl, IRegion region)
        {
            Block header = region.Header();
            if (verifier.UniformityAnalyser.IsUniform(impl.Name, header))
                return;

            var cfg = Program.GraphFromImpl(impl);
            Dictionary<Block, HashSet<Block>> controlDependence = cfg.ControlDependence();
            controlDependence.TransitiveClosure();
            cfg.ComputeLoops();
            var loopNodes = cfg.BackEdgeNodes(header).Select(item => cfg.NaturalLoops(header, item)).SelectMany(item => item);

            Expr guardEnclosingLoop = null;
            foreach (var b in controlDependence.Keys.Where(item => controlDependence[item].Contains(region.Header())))
            {
                foreach (var succ in cfg.Successors(b).Where(item => cfg.DominatorMap.DominatedBy(header, item)))
                {
                    var guard = MaybeExtractGuard(verifier, impl, succ);
                    if (guard != null)
                    {
                        guardEnclosingLoop = guardEnclosingLoop == null ? guard : Expr.And(guardEnclosingLoop, guard);
                        break;
                    }
                }
            }

            if (guardEnclosingLoop != null)
            {
                verifier.AddCandidateInvariant(
                    region,
                    Expr.Imp(Expr.Ident(verifier.FindOrCreateEnabledVariable()), guardEnclosingLoop),
                    "conditionsImpliedByEnabledness");
            }

            var cfgDual = cfg.Dual(new Block());
            Block loopConditionDominator = header;

            // The dominator might have multiple successors
            while (cfg.Successors(loopConditionDominator).Count(item => loopNodes.Contains(item)) > 1)
            {
                // Find the immediate post-dominator of the successors
                Block block = null;
                foreach (var succ in cfg.Successors(loopConditionDominator).Where(item => loopNodes.Contains(item)))
                {
                    if (block == null)
                        block = succ;
                    else
                        block = cfgDual.DominatorMap.LeastCommonAncestor(block, succ);
                }

                // Use the immediate post-dominator
                loopConditionDominator = block;
            }

            Expr guardIncludingLoopCondition = null;
            foreach (var succ in cfg.Successors(loopConditionDominator).Where(item => loopNodes.Contains(item)))
            {
                var guard = MaybeExtractGuard(verifier, impl, succ);
                if (guard != null)
                {
                    // There is at most one successor, so it's safe not use GuardIncludingLoopCondition on the rhs
                    guardIncludingLoopCondition = guardEnclosingLoop == null ? guard : Expr.And(guardEnclosingLoop, guard);
                    break;
                }
            }

            if (guardIncludingLoopCondition != null)
            {
                verifier.AddCandidateInvariant(
                    region, Expr.Imp(guardIncludingLoopCondition, Expr.Ident(verifier.FindOrCreateEnabledVariable())), "conditionsImplyingEnabledness", "do_not_predicate");
            }
        }

        private static Expr MaybeExtractGuard(GPUVerifier verifier, Implementation impl, Block b)
        {
            if (b.Cmds.Count() > 0)
            {
                var a = b.Cmds[0] as AssumeCmd;
                if (a != null && QKeyValue.FindBoolAttribute(a.Attributes, "partition"))
                {
                    if (a.Expr is IdentifierExpr)
                    {
                        return verifier.VarDefAnalysesRegion[impl].DefOfVariableName(((IdentifierExpr)a.Expr).Name);
                    }
                    else if (a.Expr is NAryExpr)
                    {
                        var nary = (NAryExpr)a.Expr;
                        if (nary.Fun is UnaryOperator
                            && (nary.Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not
                            && nary.Args[0] is IdentifierExpr)
                        {
                            var d = verifier.VarDefAnalysesRegion[impl].DefOfVariableName(((IdentifierExpr)(a.Expr as NAryExpr).Args[0]).Name);
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
            var modset = region.GetModifiedVariables().Select(x => x.Name);
            foreach (var v in partitionVars)
            {
                Expr expr = verifier.VarDefAnalysesRegion[impl].DefOfVariableName(v.Name);
                if (expr == null)
                    continue;
                var visitor = new VariablesOccurringInExpressionVisitor();
                visitor.Visit(expr);
                guardVars.UnionWith(
                    visitor.GetVariables().Where(x => x.Name.StartsWith("$")
                        && !formals.Contains(x.Name) && modset.Contains(x.Name)
                        && !verifier.UniformityAnalyser.IsUniform(impl.Name, x.Name)
                        && x.TypedIdent.Type.IsBv && (x.TypedIdent.Type.BvBits % 8 == 0)));
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
                            Function otherbv = verifier.FindOrCreateOther(sub.Type);
                            var inv = Expr.Eq(sub, new NAryExpr(Token.NoToken, new FunctionCall(otherbv), args));
                            verifier.AddCandidateInvariant(region, inv, "guardMinusInitialIsUniform");
                            var groupInv = Expr.Imp(verifier.ThreadsInSameGroup(), inv);
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
            var modset = region.GetModifiedVariables().Select(x => x.Name);
            Regex pattern = new Regex(@"\bBV\d*_((SLE)|(SLT)|(SGE)|(SGT))\b");
            foreach (var v in partitionVars)
            {
                var expr = verifier.VarDefAnalysesRegion[impl].DefOfVariableName(v.Name);
                if (!(expr is NAryExpr))
                    continue;
                var nary = expr as NAryExpr;
                if (!pattern.Match(nary.Fun.FunctionName).Success)
                    continue;
                var visitor = new VariablesOccurringInExpressionVisitor();
                visitor.Visit(nary);
                nonnegVars.UnionWith(
                    visitor.GetVariables().Where(x => x.Name.StartsWith("$")
                        && !formals.Contains(x.Name) && modset.Contains(x.Name)
                        && x.TypedIdent.Type.IsBv));
            }

            foreach (var v in nonnegVars)
            {
                // REVISIT: really we only want to guess for /integer/ variables.
                var type = v.TypedIdent.Type;
                if (type.BvBits >= 8)
                {
                    var inv = verifier.IntRep.MakeSle(verifier.IntRep.GetZero(type), new IdentifierExpr(v.tok, v));
                    verifier.AddCandidateInvariant(region, inv, "guardNonNeg");
                }
            }
        }

        private static void GenerateCandidateForReducedStrengthStrideVariables(GPUVerifier verifier, Implementation impl, IRegion region)
        {
            var rsa = verifier.ReducedStrengthAnalysesRegion[impl];
            var regionId = region.Identifier();
            foreach (string iv in rsa.StridedInductionVariables(regionId))
            {
                var sc = rsa.GetStrideConstraint(iv, regionId);
                Variable ivVariable = impl.LocVars.Where(item => item.Name == iv).First();
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
            HashSet<Variable> modifiedVariables = region.GetModifiedVariables();

            // Get the partition variables associated with the header
            HashSet<Variable> partitionVars = region.PartitionVariablesOfRegion();
            foreach (Variable v in partitionVars)
            {
                // Find the expression which defines a particular partition variable.
                // Visit the expression and select any variable in the mod set of the loop.
                // We assume that any variable satisfying these conditions is a loop counter
                Expr partitionDefExpr = verifier.VarDefAnalysesRegion[impl].DefOfVariableName(v.Name);
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
                        foreach (var lhsRhs in lhss.Zip(cmd.Rhss))
                        {
                            if (lhsRhs.Item1.DeepAssignedVariable.Name == loopCounter.Name)
                            {
                                verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSle(new IdentifierExpr(loopCounter.tok, loopCounter), lhsRhs.Item2), "loopBound");
                                verifier.AddCandidateInvariant(region, verifier.IntRep.MakeSge(new IdentifierExpr(loopCounter.tok, loopCounter), lhsRhs.Item2), "loopBound");
                                verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUle(new IdentifierExpr(loopCounter.tok, loopCounter), lhsRhs.Item2), "loopBound");
                                verifier.AddCandidateInvariant(region, verifier.IntRep.MakeUge(new IdentifierExpr(loopCounter.tok, loopCounter), lhsRhs.Item2), "loopBound");
                            }
                        }
                    }
                }
            }
        }

        public static void PostInstrument(GPUVerifier verifier, Implementation impl)
        {
            new LoopInvariantGenerator(verifier, impl).PostInstrument();
        }

        private void PostInstrument()
        {
            HashSet<Variable> localVars = new HashSet<Variable>();
            foreach (Variable v in impl.LocVars)
                localVars.Add(v);

            foreach (Variable v in impl.InParams)
                localVars.Add(v);

            foreach (Variable v in impl.OutParams)
                localVars.Add(v);

            AddCandidateInvariants(localVars, impl);
        }

        private void AddPredicatedEqualityCandidateInvariant(IRegion region, string loopPredicate, Variable v)
        {
            var inv = Expr.Imp(
             Expr.And(
              new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, loopPredicate + "$1", Microsoft.Boogie.Type.Int))),
              new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, loopPredicate + "$2", Microsoft.Boogie.Type.Int)))),
             Expr.Eq(
              new IdentifierExpr(Token.NoToken, new VariableDualiser(1, verifier, impl.Name).VisitVariable(v.Clone() as Variable)),
              new IdentifierExpr(Token.NoToken, new VariableDualiser(2, verifier, impl.Name).VisitVariable(v.Clone() as Variable))));

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
                            var v = Utilities.StripThreadIdentifier(
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

        private void AddBarrierDivergenceCandidates(HashSet<Variable> localVars, Implementation impl, IRegion region)
        {
            if (!verifier.ContainsBarrierCall(region) && !GPUVerifyVCGenCommandLineOptions.WarpSync)
                return;

            Expr guard = region.Guard();
            if (guard != null && verifier.UniformityAnalyser.IsUniform(impl.Name, guard))
                return;

            if (IsDisjunctionOfPredicates(guard))
            {
                string loopPredicate = ((guard as NAryExpr).Args[0] as IdentifierExpr).Name;
                loopPredicate = loopPredicate.Substring(0, loopPredicate.IndexOf('$'));

                // Int type used here, but it doesn't matter as we will print and then re-parse the program
                var uniformEnabledPredicate = Expr.Eq(
                    new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, loopPredicate + "$1", Type.Int))),
                    new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, loopPredicate + "$2", Type.Int))));

                verifier.AddCandidateInvariant(region, uniformEnabledPredicate, "loopPredicateEquality");

                verifier.AddCandidateInvariant(region, Expr.Imp(verifier.ThreadsInSameGroup(), uniformEnabledPredicate), "loopPredicateEquality");

                Dictionary<string, int> assignmentCounts = GetAssignmentCounts(impl);

                HashSet<string> alreadyConsidered = new HashSet<string>();

                foreach (var v in localVars)
                {
                    string lv = Utilities.StripThreadIdentifier(v.Name);
                    if (alreadyConsidered.Contains(lv))
                        continue;

                    alreadyConsidered.Add(lv);

                    if (verifier.UniformityAnalyser.IsUniform(impl.Name, v.Name))
                        continue;

                    if (GPUVerifier.IsPredicate(lv))
                        continue;

                    if (!assignmentCounts.ContainsKey(lv) || assignmentCounts[lv] <= 1)
                        continue;

                    if (!verifier.ContainsNamedVariable(region.GetModifiedVariables(), lv))
                        continue;

                    AddPredicatedEqualityCandidateInvariant(region, loopPredicate, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, lv, Type.Int)));
                }
            }
        }

        private static bool IsDisjunctionOfPredicates(Expr guard)
        {
            NAryExpr nary = guard as NAryExpr;
            if (nary == null || nary.Args.Count() != 2)
                return false;

            BinaryOperator binOp = nary.Fun as BinaryOperator;
            if (binOp == null || binOp.Op != BinaryOperator.Opcode.Or)
                return false;

            if (!(nary.Args[0] is IdentifierExpr && nary.Args[1] is IdentifierExpr))
                return false;

            return GPUVerifier.IsPredicate(Utilities.StripThreadIdentifier(((IdentifierExpr)nary.Args[0]).Name))
                && GPUVerifier.IsPredicate(Utilities.StripThreadIdentifier(((IdentifierExpr)nary.Args[1]).Name));
        }

        private void AddCandidateInvariants(HashSet<Variable> localVars, Implementation impl)
        {
            foreach (IRegion region in verifier.RootRegion(impl).SubRegions())
            {
                if (verifier.RegionHasLoopInvariantsDisabled(region))
                    continue;

                foreach (InvariantGenerationRule r in invariantGenerationRules)
                    r.GenerateCandidates(impl, region);

                AddBarrierDivergenceCandidates(localVars, impl, region);

                verifier.RaceInstrumenter.AddRaceCheckingCandidateInvariants(impl, region);
            }
        }

        private static bool AccessesGlobalArrayOrUnsafeBarrier(Cmd c, GPUVerifier verifier)
        {
            var stateToCheck = verifier.KernelArrayInfo;

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
                if (GPUVerifier.IsBarrier(call.Proc)
                    && !QKeyValue.FindBoolAttribute(call.Proc.Attributes, "safe_barrier"))
                {
                    return true;
                }

                // Speculate invariants if we see a call to a procedure that has a non-local array
                // or constant array in its modset
                List<Variable> vars = new List<Variable>();
                call.AddAssignedVariables(vars);
                foreach (Variable v in vars)
                {
                    if (stateToCheck.GetGlobalAndGroupSharedArrays(false).Contains(v))
                        return true;

                    if (stateToCheck.GetConstantArrays().Contains(v))
                        return true;
                }
            }

            // Speculate invariants if race instrumentation or a constant write
            // instrumentation will occur
            if (c is AssignCmd)
            {
                AssignCmd assign = c as AssignCmd;

                ReadCollector rc = new ReadCollector(stateToCheck);
                foreach (var rhs in assign.Rhss)
                    rc.Visit(rhs);
                foreach (var access in rc.NonPrivateAccesses)
                {
                    // Ignore disabled arrays
                    if (stateToCheck.GetGlobalAndGroupSharedArrays(false).Contains(access.V))
                    {
                        // Ignore read-only arrays (whether or not they are disabled)
                        if (!stateToCheck.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(access.V))
                            return true;
                    }
                }

                foreach (var lhsRhs in assign.Lhss.Zip(assign.Rhss))
                {
                    WriteCollector wc = new WriteCollector(stateToCheck);
                    wc.Visit(lhsRhs.Item1);
                    if (wc.FoundNonPrivateWrite())
                    {
                        // Ignore disabled arrays
                        if (stateToCheck.GetGlobalAndGroupSharedArrays(false).Contains(wc.GetAccess().V))
                            return true;
                    }
                }

                foreach (var lhsRhs in assign.Lhss.Zip(assign.Rhss))
                {
                    ConstantWriteCollector cwc = new ConstantWriteCollector(stateToCheck);
                    cwc.Visit(lhsRhs.Item1);
                    if (cwc.FoundWrite())
                    {
                        // Ignore disabled arrays
                        if (stateToCheck.GetGlobalAndGroupSharedArrays(false).Contains(cwc.GetAccess().V))
                            return true;
                    }
                }
            }

            // Speculate invariants if we see an assert that is not a sourceloc or
            // block_sourceloc assert; such asserts is likely user supplied.
            if (c is AssertCmd)
            {
                AssertCmd assertion = c as AssertCmd;
                if (!QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc")
                    && !QKeyValue.FindBoolAttribute(assertion.Attributes, "block_sourceloc")
                    && !assertion.Expr.Equals(Expr.True))
                {
                    return true;
                }
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

        private static bool AccessesGlobalArrayOrUnsafeBarrier(IRegion region, GPUVerifier verifier)
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
