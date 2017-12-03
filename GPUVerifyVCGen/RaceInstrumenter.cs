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
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.Boogie;

    internal abstract class RaceInstrumenter : IRaceInstrumenter
    {
        internal GPUVerifier verifier;

        internal int CheckStateCounter = 0;

        private Dictionary<string, Procedure> raceCheckingProcedures = new Dictionary<string, Procedure>();

        public RaceInstrumenter(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public void AddDefaultLoopInvariants()
        {
            // Here we add invariants that are guaranteed to be true
            // by construction.
            foreach (IRegion region in verifier.Program.Implementations
                .Select(item => verifier.RootRegion(item).SubRegions()).SelectMany(item => item))
            {
                foreach (var a in verifier.KernelArrayInfo.GetGroupSharedArrays(false)
                    .Where(item => !verifier.KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(false).Contains(item)))
                {
                    foreach (var access in AccessType.Types)
                    {
                        region.AddInvariant(new AssertCmd(Token.NoToken,
                          Expr.Imp(AccessHasOccurredExpr(a, access), GPUVerifier.ThreadsInSameGroup()),
                          new QKeyValue(Token.NoToken, "tag", new List<object> { "groupSharedArraysDisjointAcrossGroups" }, null)));
                    }
                }
            }
        }

        public void AddDefaultContracts()
        {
            // Here we add pre- and post-conditions that are guaranteed to be true
            // by construction.
            foreach (Procedure proc in verifier.Program.TopLevelDeclarations.OfType<Procedure>()
              .Where(item => !verifier.ProcedureIsInlined(item) &&
                            !verifier.IsKernelProcedure(item) &&
                            (!verifier.ProcedureHasNoImplementation(item) || item.Modifies.Count() > 0)))
            {
                foreach (var a in verifier.KernelArrayInfo.GetGroupSharedArrays(false)
                    .Where(item => !verifier.KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(false).Contains(item)))
                {
                    foreach (var access in AccessType.Types)
                    {
                        proc.Requires.Add(new Requires(false, Expr.Imp(AccessHasOccurredExpr(a, access), GPUVerifier.ThreadsInSameGroup())));
                        proc.Ensures.Add(new Ensures(false, Expr.Imp(AccessHasOccurredExpr(a, access), GPUVerifier.ThreadsInSameGroup())));
                    }
                }
            }
        }

        private void AddNoAccessCandidateInvariants(IRegion region, Variable v)
        {
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

            if (verifier.ContainsBarrierCall(region))
            {
                foreach (var kind in AccessType.Types)
                {
                    if (verifier.ContainsNamedVariable(
                        region.GetModifiedVariables(), RaceInstrumentationUtil.MakeHasOccurredVariableName(v.Name, kind)))
                    {
                        AddNoAccessCandidateInvariant(region, v, kind);
                    }
                }
            }
        }

        private void AddNoAccessCandidateRequires(Procedure pro, Variable v)
        {
            foreach (var kind in AccessType.Types)
                AddNoAccessCandidateRequires(pro, v, kind);
        }

        private void AddNoAccessCandidateEnsures(Procedure proc, Variable v)
        {
            foreach (var kind in AccessType.Types)
                AddNoAccessCandidateEnsures(proc, v, kind);
        }

        private void AddNoAccessCandidateInvariant(IRegion region, Variable v, AccessType access)
        {
            Expr candidate = NoAccessHasOccurredExpr(v, access);
            verifier.AddCandidateInvariant(region, candidate, "no" + access.ToString().ToLower());
        }

        private void AddSameWarpNoAccessCandidateInvariant(IRegion region, Variable v, AccessType access)
        {
            if (!GPUVerifyVCGenCommandLineOptions.WarpSync)
                return;

            Expr candidate = Expr.Imp(Expr.And(GPUVerifier.ThreadsInSameGroup(), verifier.ThreadsInSameWarp()), NoAccessHasOccurredExpr(v, access));
            verifier.AddCandidateInvariant(region, candidate, "sameWarpNoAccess", "do_not_predicate");
        }

        public void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region)
        {
            List<Expr> offsetPredicatesRead = new List<Expr>();
            List<Expr> offsetPredicatesWrite = new List<Expr>();
            List<Expr> offsetPredicatesAtomic = new List<Expr>();

            foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false))
            {
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
                AddDisabledMaintainsInstrumentation(impl, region, v, AccessType.READ);
                AddDisabledMaintainsInstrumentation(impl, region, v, AccessType.WRITE);
                AddDisabledMaintainsInstrumentation(impl, region, v, AccessType.ATOMIC);
            }
        }

        private void AddDisabledMaintainsInstrumentation(Implementation impl, IRegion region, Variable v, AccessType Access)
        {
            if (!verifier.ContainsNamedVariable(region.GetModifiedVariables(), RaceInstrumentationUtil.MakeHasOccurredVariableName(v.Name, Access)))
                return;

            string dominatorPredicate = null;
            foreach (Cmd c in region.Header().Cmds)
            {
                AssumeCmd aCmd = c as AssumeCmd;
                if (aCmd != null)
                {
                    dominatorPredicate = QKeyValue.FindStringAttribute(aCmd.Attributes, "dominator_predicate");
                    if (dominatorPredicate != null)
                        break;
                }
            }

            if (dominatorPredicate == null)
                return;

            var regionName = region.Header().ToString();
            IdentifierExpr ghostAccess = new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredGhostVariable(v.Name, regionName, Access, impl));
            IdentifierExpr access = new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(v.Name, Access));
            IdentifierExpr ghostOffset = null;
            IdentifierExpr offset = null;
            if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
            {
                var ghostOffsetVar = verifier.FindOrCreateOffsetGhostVariable(v.Name, regionName, Access, impl);
                ghostOffset = new IdentifierExpr(Token.NoToken, ghostOffsetVar);
                offset = new IdentifierExpr(Token.NoToken, verifier.FindOrCreateOffsetVariable(v.Name, Access));
            }

            foreach (Block b in region.PreHeaders())
            {
                b.Cmds.Add(Cmd.SimpleAssign(Token.NoToken, ghostAccess, access));
                if (ghostOffset != null)
                    b.Cmds.Add(Cmd.SimpleAssign(Token.NoToken, ghostOffset, offset));
            }

            Variable dominator = new LocalVariable(Token.NoToken,
                                                new TypedIdent(Token.NoToken, dominatorPredicate, Microsoft.Boogie.Type.Bool));
            Expr dominatorExpr = verifier.MaybeDualise(new IdentifierExpr(Token.NoToken, dominator), 1, impl.Name);
            Expr eqAccess = Expr.Eq(ghostAccess, access);
            verifier.AddCandidateInvariant(region, Expr.Imp(Expr.Not(dominatorExpr), eqAccess), "disabledMaintainsInstrumentation");
            if (ghostOffset != null)
            {
                Expr eqOffset = Expr.Eq(ghostOffset, offset);
                verifier.AddCandidateInvariant(region, Expr.Imp(Expr.Not(dominatorExpr), eqOffset), "disabledMaintainsInstrumentation");
            }

            // Could speculate similar invariant for benign instrumentation variable
        }

        /*
         * Generates candidate invariants by rewriting offset expressions.
         *
         * A component is an identifier (i.e., local_id_{x,y,z} or group_id_{x,y,z})
         * where /at least one/ component is necessarily distinct between distinct
         * threads. Given an offset expression we extract components using division
         * and subtraction.
         */
        private void AddComponentBreakingCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType access)
        {
            // get offset expression
            // continue if there is exactly one offset expression, or,
            // if all offsets are to the same logical element of a vector type (e.g., uint2).
            HashSet<Expr> offsets = GetOffsetsAccessed(region, v, access);
            if (offsets.Count() == 0)
                return;

            if (offsets.Count() > 1)
            {
                HashSet<string> vs = new HashSet<string>();
                foreach (var offset in offsets)
                {
                    var visitor = new VariablesOccurringInExpressionVisitor();
                    visitor.Visit(offset);
                    vs.UnionWith(visitor.GetVariables().Select(x => x.Name));
                    // Could further refine by checking offset has form
                    // BV32_ADD(BV32_MUL(v, c+i) where c is a constant 2,3,4
                    //                           and   i is in [0,c)
                }

                if (vs.Count() != 1)
                    return;
            }

            // now get the offset definition, distribute and try breaking apart
            bool isConstant;
            bool isSubstitutable;
            var def = verifier.varDefAnalysesRegion[impl].SubstDefinitions(offsets.First(), impl.Name, out isConstant, out isSubstitutable);
            if (isConstant || !isSubstitutable)
                return;

            // Should also check expression consists only of adds and muls?
            var distribute = new DistributeExprVisitor(verifier);
            var rewrite = distribute.Visit(def);
            var component = new ComponentVisitor();
            component.Visit(rewrite);
            var invariants = component.GenerateCandidateInvariants(verifier, v, access);
            foreach (Expr inv in invariants)
            {
                verifier.AddCandidateInvariant(region, inv, "accessBreak");
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowAccessBreaking)
            {
                Console.WriteLine("Access breaking of [{0}]", def);
                component.Dump();
            }
        }

        class FunctionsOccurringInExpressionVisitor : StandardVisitor
        {
            private HashSet<string> functions = new HashSet<string>();

            internal IEnumerable<string> GetFunctions()
            {
                return functions;
            }

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                functions.Add(node.Fun.FunctionName);
                return base.VisitNAryExpr(node);
            }
        }

        class VariablesOrLiteralsOccurringInExpressionVisitor : StandardVisitor
        {
            private HashSet<Expr> terms = new HashSet<Expr>();

            internal IEnumerable<Expr> GetVariablesOrLiterals()
            {
                return terms;
            }

            public override Variable VisitVariable(Variable node)
            {
                terms.Add(new IdentifierExpr(Token.NoToken, node));
                return base.VisitVariable(node);
            }

            public override Expr VisitLiteralExpr(LiteralExpr node)
            {
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
        class ComponentVisitor : StandardVisitor
        {
            private HashSet<string> allComponents;
            private Dictionary<Expr, HashSet<Expr>> componentMap;
            private bool canAccessBreak;

            internal ComponentVisitor()
            {
                componentMap = new Dictionary<Expr, HashSet<Expr>>();
                allComponents = new HashSet<string>();
                allComponents.Add(GPUVerifier.MakeThreadId("X", 1).Name);
                allComponents.Add(GPUVerifier.MakeThreadId("Y", 1).Name);
                allComponents.Add(GPUVerifier.MakeThreadId("Z", 1).Name);
                allComponents.Add(GPUVerifier.MakeGroupId("X", 1).Name);
                allComponents.Add(GPUVerifier.MakeGroupId("Y", 1).Name);
                allComponents.Add(GPUVerifier.MakeGroupId("Z", 1).Name);
                canAccessBreak = true;
            }

            public void Dump()
            {
                if (canAccessBreak)
                {
                    Console.WriteLine("Can access break expression.");
                    foreach (var c in componentMap.Keys)
                    {
                        var terms = componentMap[c];
                        Console.WriteLine("Component {0} has {1} multiply terms", c, terms.Count());
                        foreach (var t in terms)
                        {
                            Console.WriteLine("  Term {0}", t);
                        }
                    }
                }
                else
                {
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
            public IEnumerable<Expr> GenerateCandidateInvariants(GPUVerifier verifier, Variable v, AccessType access)
            {
                // Mixing of ids of type bv32 and offsets of type bv64 is
                // possible with CUDA, and incompatible with the current code.
                if (verifier.size_t_bits == 64)
                    return Enumerable.Empty<Expr>();
                if (!canAccessBreak)
                    return Enumerable.Empty<Expr>();

                var result = new List<Expr>();
                var offsetVar = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits));
                foreach (var c in componentMap.Keys)
                {
                    Expr invariant = new IdentifierExpr(Token.NoToken, offsetVar);
                    var xs = componentMap[c];
                    foreach (var x in xs)
                    {
                        invariant = verifier.IntRep.MakeDiv(invariant, x);
                    }

                    foreach (var d in componentMap.Keys.Where(x => x != c))
                    {
                        Expr subexpr = d;
                        var ys = componentMap[d];
                        foreach (var y in ys.Except(xs))
                        {
                            subexpr = verifier.IntRep.MakeMul(subexpr, y);
                        }

                        foreach (var x in xs.Except(ys))
                        {
                            subexpr = verifier.IntRep.MakeDiv(subexpr, x);
                        }

                        invariant = verifier.IntRep.MakeSub(invariant, subexpr);
                    }

                    invariant = Expr.Eq(c, invariant);
                    invariant = Expr.Imp(new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)), invariant);
                    result.Add(invariant);
                }

                return result;
            }

            private bool IsMultiplyExpr(Expr e)
            {
                var visitor = new FunctionsOccurringInExpressionVisitor();
                visitor.Visit(e);
                var fs = visitor.GetFunctions();
                return fs.Count() == 1 && fs.Single() == "BV32_MUL";
            }

            /*
             * Extract information for each subexpression containing only multiplies
             * of a single component and terms (literals or variables) and add it to
             * the ComponentMap. We conservatively bail by unsetting CanAccessBreak.
             *
             * TODO: do not ignore guard expression
             * TODO: rewrite for arbitrary bitwidths (currently fixed for BV32)
             */
            public override Expr VisitNAryExpr(NAryExpr node)
            {
                if (node.Fun.FunctionName.Equals("BV32_MUL"))
                {
                    if (IsMultiplyExpr(node))
                    {
                        var visitor = new VariablesOrLiteralsOccurringInExpressionVisitor();
                        visitor.Visit(node);
                        var terms = visitor.GetVariablesOrLiterals();
                        var components = terms.Where(t => t is IdentifierExpr && allComponents.Contains((t as IdentifierExpr).Decl.Name));
                        if (components.Count() == 0)
                        { // assume guard expression
                            return node;
                        }
                        else if (components.Count() == 1)
                        {
                            if (terms.All(t => (t.Type as BvType).Bits == 32))
                            {
                                var c = components.Single();
                                var termsExceptC = new HashSet<Expr>(terms.Where(t => t is LiteralExpr || t != c));
                                if (!componentMap.ContainsKey(c))
                                {
                                    componentMap[c] = termsExceptC;
                                    return node;
                                }
                            }
                        }
                    }

                    // otherwise bail
                    canAccessBreak = false;
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
        class DistributeExprVisitor : Duplicator
        {
            private GPUVerifier verifier;

            internal DistributeExprVisitor(GPUVerifier verifier)
            {
                this.verifier = verifier;
            }

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                if (node.Fun.FunctionName.Equals("BV32_MUL"))
                {
                    Expr lhs = node.Args[0];
                    Expr rhs = node.Args[1];
                    bool lhsIsLeaf = lhs is LiteralExpr || lhs is IdentifierExpr;
                    bool rhsIsLeaf = rhs is LiteralExpr || rhs is IdentifierExpr;
                    if (!(lhsIsLeaf && rhsIsLeaf))
                    {
                        Expr leaf = lhsIsLeaf ? lhs : rhs;
                        NAryExpr term = (lhsIsLeaf ? rhs : lhs) as NAryExpr;
                        if (term != null && term.Fun.FunctionName.Equals("BV32_ADD"))
                        {
                            Expr e1 = term.Args[0];
                            Expr e2 = term.Args[1];
                            Expr newLhs = verifier.IntRep.MakeMul(leaf, e1);
                            Expr newRhs = verifier.IntRep.MakeMul(leaf, e2);
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

        private bool DoesNotReferTo(Expr expr, string v)
        {
            FindReferencesToNamedVariableVisitor visitor = new FindReferencesToNamedVariableVisitor(v);
            visitor.VisitExpr(expr);
            return !visitor.found;
        }

        private int ParameterOffsetForSource(AccessType access)
        {
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
            {
                return 4;
            }
            else if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.READ)
            {
                return 3;
            }
            else
            {
                return 2;
            }
        }

        private List<Expr> CollectOffsetPredicates(Implementation impl, IRegion region, Variable v, AccessType access)
        {
            var offsetVar = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits));
            var offsetExpr = new IdentifierExpr(Token.NoToken, offsetVar);
            var offsetPreds = new List<Expr>();

            foreach (var offset in GetOffsetsAccessed(region, v, access))
            {
                bool isConstant;
                bool isSubstitutable;
                var def = verifier.varDefAnalysesRegion[impl].SubstDefinitions(offset, impl.Name, out isConstant, out isSubstitutable);
                if (!isSubstitutable)
                    continue;
                if (isConstant)
                {
                    offsetPreds.Add(Expr.Eq(offsetExpr, def));
                }
                else
                {
                    var sc = StrideConstraint.FromExpr(verifier, impl, def);
                    var pred = sc.MaybeBuildPredicate(verifier, offsetExpr);
                    if (pred != null)
                        offsetPreds.Add(pred);
                }
            }

            return offsetPreds;
        }

        private void AddOffsetIsBlockBoundedCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType access)
        {
            var modset = region.GetModifiedVariables().Select(x => x.Name);
            foreach (Expr e in GetOffsetsAccessed(region, v, access))
            {
                NAryExpr nary = e as NAryExpr;
                if (nary == null || !nary.Fun.FunctionName.Equals("BV32_ADD"))
                    continue;

                Expr lhs = nary.Args[0];
                Expr rhs = nary.Args[1];
                var lhsVisitor = new VariablesOccurringInExpressionVisitor();
                lhsVisitor.Visit(lhs);
                var lhsVars = lhsVisitor.GetVariables();
                var rhsVisitor = new VariablesOccurringInExpressionVisitor();
                rhsVisitor.Visit(rhs);
                var rhsVars = rhsVisitor.GetVariables();
                Expr constant;

                if (lhsVars.All(x => !modset.Contains(x.Name)) && rhsVars.Any(x => modset.Contains(x.Name)))
                    constant = lhs;
                else if (rhsVars.All(x => !modset.Contains(x.Name)) && lhsVars.Any(x => modset.Contains(x.Name)))
                    constant = rhs;
                else
                    continue;

                bool isConstant;
                bool isSubstitutable;
                Expr lowerBound = verifier.varDefAnalysesRegion[impl].SubstDefinitions(constant, impl.Name, out isConstant, out isSubstitutable);
                if (!isSubstitutable)
                    continue;

                var visitor = new VariablesOccurringInExpressionVisitor();
                visitor.VisitExpr(lowerBound);
                var groupIds = visitor.GetVariables().Where(x => GPUVerifier.IsDualisedGroupIdConstant(x));
                if (groupIds.Count() != 1)
                    continue;

                // Getting here means the access consists of a constant (not in the
                // loop's modset) plus a changing index. Furthermore, the constant
                // contains exactly one group-id variable. We guess this forms a lower
                // and upper bound for the access. i.e.,
                //   constant <= access <= constant[group-id+1/group-id]
                Variable groupId = groupIds.Single();
                Expr groupIdPlusOne = verifier.IntRep.MakeAdd(new IdentifierExpr(Token.NoToken, groupId), verifier.IntRep.GetLiteral(1, verifier.id_size_bits));
                Dictionary<Variable, Expr> substs = new Dictionary<Variable, Expr>();
                substs.Add(groupId, groupIdPlusOne);
                Substitution s = Substituter.SubstitutionFromHashtable(substs);
                Expr upperBound = Substituter.Apply(s, lowerBound);
                var lowerBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access), verifier.IntRep.MakeSle(lowerBound, OffsetXExpr(v, access, 1)));
                var upperBoundInv = Expr.Imp(GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access), verifier.IntRep.MakeSlt(OffsetXExpr(v, access, 1), upperBound));
                verifier.AddCandidateInvariant(region, lowerBoundInv, "accessLowerBoundBlock");
                verifier.AddCandidateInvariant(region, upperBoundInv, "accessUpperBoundBlock");
            }
        }

        private void AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(Implementation impl, IRegion region, Variable v, AccessType access)
        {
            KeyValuePair<IdentifierExpr, Expr> iLessThanC = GetILessThanC(region.Guard(), impl);
            if (iLessThanC.Key != null)
            {
                foreach (Expr e in GetOffsetsAccessed(region, v, access))
                {
                    if (HasFormIPlusLocalIdTimesC(e, iLessThanC, impl))
                    {
                        AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(region, v, iLessThanC.Value, access);
                        break;
                    }
                }

                foreach (Expr e in GetOffsetsAccessed(region, v, access))
                {
                    if (HasFormIPlusGlobalIdTimesC(e, iLessThanC, impl))
                    {
                        AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(region, v, iLessThanC.Value, access);
                        break;
                    }
                }
            }
        }

        private bool HasFormIPlusLocalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl)
        {
            NAryExpr nary = e as NAryExpr;
            if (nary == null || !nary.Fun.FunctionName.Equals("BV32_ADD"))
                return false;

            return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
                IsLocalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
                (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
                IsLocalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
        }

        private bool IsLocalIdTimesConstant(Expr maybeLocalIdTimesConstant, Expr constant, Implementation impl)
        {
            NAryExpr nary = maybeLocalIdTimesConstant as NAryExpr;
            if (nary == null || !nary.Fun.FunctionName.Equals("BV32_MUL"))
                return false;

            return
                (SameConstant(nary.Args[0], constant) && verifier.IsLocalId(nary.Args[1], 0, impl)) ||
                (SameConstant(nary.Args[1], constant) && verifier.IsLocalId(nary.Args[0], 0, impl));
        }

        private bool HasFormIPlusGlobalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl)
        {
            NAryExpr nary = e as NAryExpr;
            if (nary == null || !nary.Fun.FunctionName.Equals("BV32_ADD"))
                return false;

            return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
                IsGlobalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
                (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
                IsGlobalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
        }

        private bool IsGlobalIdTimesConstant(Expr maybeGlobalIdTimesConstant, Expr constant, Implementation impl)
        {
            NAryExpr nary = maybeGlobalIdTimesConstant as NAryExpr;
            if (nary == null || !nary.Fun.FunctionName.Equals("BV32_MUL"))
                return false;

            return
                (SameConstant(nary.Args[0], constant) && verifier.IsGlobalId(nary.Args[1], 0, impl)) ||
                (SameConstant(nary.Args[1], constant) && verifier.IsGlobalId(nary.Args[0], 0, impl));
        }

        private bool SameConstant(Expr expr, Expr constant)
        {
            if (constant is IdentifierExpr)
            {
                IdentifierExpr identifierExpr = constant as IdentifierExpr;
                Debug.Assert(identifierExpr.Decl is Constant);
                return expr is IdentifierExpr && (expr as IdentifierExpr).Decl is Constant && (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Decl.Name);
            }
            else
            {
                Debug.Assert(constant is LiteralExpr);
                LiteralExpr literalExpr = constant as LiteralExpr;
                if (!(expr is LiteralExpr))
                {
                    return false;
                }

                if (!(literalExpr.Val is BvConst) || !((expr as LiteralExpr).Val is BvConst))
                {
                    return false;
                }

                return (literalExpr.Val as BvConst).Value.ToInt == ((expr as LiteralExpr).Val as BvConst).Value.ToInt;
            }
        }

        private bool SameIdentifierExpression(Expr expr, IdentifierExpr identifierExpr)
        {
            if (!(expr is IdentifierExpr))
            {
                return false;
            }

            return (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Name);
        }

        private KeyValuePair<IdentifierExpr, Expr> GetILessThanC(Expr expr, Implementation impl)
        {
            bool guardHasOuterNot = false;
            if (expr is NAryExpr &&
                (expr as NAryExpr).Fun is BinaryOperator &&
                ((expr as NAryExpr).Fun as BinaryOperator).Op == BinaryOperator.Opcode.And)
            {
                Expr lhs = (expr as NAryExpr).Args[0];
                Expr rhs = (expr as NAryExpr).Args[1];

                // !v && !v
                if (lhs is NAryExpr &&
                      (lhs as NAryExpr).Fun is UnaryOperator &&
                      ((lhs as NAryExpr).Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not &&
                    rhs is NAryExpr &&
                      (rhs as NAryExpr).Fun is UnaryOperator &&
                      ((rhs as NAryExpr).Fun as UnaryOperator).Op == UnaryOperator.Opcode.Not)
                {
                    lhs = (lhs as NAryExpr).Args[0];
                    rhs = (rhs as NAryExpr).Args[0];
                    guardHasOuterNot = true;
                }

                if (lhs is IdentifierExpr && rhs is IdentifierExpr)
                {
                    Variable lhsVar = (lhs as IdentifierExpr).Decl;
                    Variable rhsVar = (rhs as IdentifierExpr).Decl;
                    if (lhsVar.Name == rhsVar.Name)
                    {
                        expr = verifier.varDefAnalysesRegion[impl].DefOfVariableName(lhsVar.Name);
                    }
                }
            }

            if (!(expr is NAryExpr))
            {
                return new KeyValuePair<IdentifierExpr, Expr>(null, null);
            }

            NAryExpr nary = expr as NAryExpr;

            if (!guardHasOuterNot)
            {
                if (!(nary.Fun.FunctionName.Equals("BV32_ULT") ||
                      nary.Fun.FunctionName.Equals("BV32_SLT")))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                if (!(nary.Args[0] is IdentifierExpr))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                if (!IsConstant(nary.Args[1]))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                return new KeyValuePair<IdentifierExpr, Expr>(nary.Args[0] as IdentifierExpr, nary.Args[1]);
            }
            else
            {
                if (!(nary.Fun.FunctionName.Equals("BV32_UGT") ||
                      nary.Fun.FunctionName.Equals("BV32_SGT")))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                if (!(nary.Args[1] is IdentifierExpr))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                if (!IsConstant(nary.Args[0]))
                {
                    return new KeyValuePair<IdentifierExpr, Expr>(null, null);
                }

                return new KeyValuePair<IdentifierExpr, Expr>(nary.Args[1] as IdentifierExpr, nary.Args[0]);
            }
        }

        private static bool IsConstant(Expr e)
        {
            return (e is IdentifierExpr && (e as IdentifierExpr).Decl is Constant) || e is LiteralExpr;
        }

        private void AddReadOrWrittenOffsetIsThreadIdCandidateRequires(Procedure proc, Variable v)
        {
            foreach (var kind in AccessType.Types)
                AddAccessedOffsetIsThreadLocalIdCandidateRequires(proc, v, kind);
        }

        private void AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(Procedure proc, Variable v)
        {
            foreach (var kind in AccessType.Types)
                AddAccessedOffsetIsThreadLocalIdCandidateEnsures(proc, v, kind);
        }

        public void AddKernelPrecondition()
        {
            foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false))
                AddRequiresNoPendingAccess(v);
        }

        public void AddRaceCheckingInstrumentation()
        {
            foreach (var impl in verifier.Program.Implementations.ToList())
                new ImplementationInstrumenter(this, impl).AddRaceCheckCalls();
        }

        protected abstract void AddLogAccessProcedure(Variable v, AccessType access);

        private void AddRaceCheckingDecsAndProcsForVar(Variable v)
        {
            foreach (var kind in AccessType.Types)
            {
                AddLogRaceDeclarations(v, kind);
                AddLogAccessProcedure(v, kind);
                AddCheckAccessProcedure(v, kind);
            }

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
            {
                AddUpdateBenignFlagProcedure(v);
            }
        }

        internal Procedure GetRaceCheckingProcedure(IToken tok, string name)
        {
            if (raceCheckingProcedures.ContainsKey(name))
                return raceCheckingProcedures[name];

            Procedure newProcedure = new Procedure(tok, name, new List<TypeVariable>(), new List<Variable>(), new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(), new List<Ensures>());
            raceCheckingProcedures[name] = newProcedure;
            return newProcedure;
        }

        public BigBlock MakeResetReadWriteSetStatements(Variable v, Expr resetCondition)
        {
            // We only want to do this reset for enabled arrays
            Debug.Assert(verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(v));

            BigBlock result = new BigBlock(Token.NoToken, null, new List<Cmd>(), null, null);

            foreach (var kind in AccessType.Types)
            {
                Expr resetAssumeGuard = Expr.Imp(resetCondition,
                  Expr.Not(Expr.Ident(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, kind))));

                if (verifier.KernelArrayInfo.GetGlobalArrays(false).Contains(v))
                    resetAssumeGuard = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), resetAssumeGuard);

                if (new AccessType[] { AccessType.READ, AccessType.WRITE }.Contains(kind)
                  && verifier.ArraysAccessedByAsyncWorkGroupCopy[kind].Contains(v.Name))
                {
                    resetAssumeGuard = Expr.Imp(
                      Expr.Eq(Expr.Ident(verifier.FindOrCreateAsyncHandleVariable(v.Name, kind)), verifier.FindOrCreateAsyncNoHandleConstant()),
                      resetAssumeGuard);
                }

                result.simpleCmds.Add(new AssumeCmd(Token.NoToken, resetAssumeGuard));
            }

            return result;
        }

        protected Procedure MakeLogAccessProcedureHeader(Variable v, AccessType access)
        {
            // We only want to make this header for enabled arrays
            Debug.Assert(verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(v));

            List<Variable> inParams = new List<Variable>();

            Variable predicateParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_P", Microsoft.Boogie.Type.Bool));

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);
            Variable offsetParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_offset", mt.Arguments[0]));
            Variable valueParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value", mt.Result));
            Variable valueOldParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value_old", mt.Result));

            Variable asyncHandleParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_async_handle", verifier.IntRep.GetIntType(verifier.size_t_bits)));

            Debug.Assert(!(mt.Result is MapType));

            inParams.Add(predicateParameter);
            inParams.Add(offsetParameter);
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
                inParams.Add(valueParameter);

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
                inParams.Add(valueOldParameter);

            if ((access == AccessType.READ || access == AccessType.WRITE) &&
              verifier.ArraysAccessedByAsyncWorkGroupCopy[access].Contains(v.Name))
            {
                inParams.Add(asyncHandleParameter);
            }

            string logProcedureName = "_LOG_" + access + "_" + v.Name;

            Procedure result = GetRaceCheckingProcedure(v.tok, logProcedureName);

            result.InParams = inParams;

            GPUVerifier.AddInlineAttribute(result);

            return result;
        }

        protected Procedure MakeUpdateBenignFlagProcedureHeader(Variable v)
        {
            // We only want to make this header for enabled arrays
            Debug.Assert(verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(v));

            List<Variable> inParams = new List<Variable>();

            Variable predicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);
            Variable offsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
            Debug.Assert(!(mt.Result is MapType));

            inParams.Add(predicateParameter);
            inParams.Add(offsetParameter);

            string updateBenignFlagProcedureName = "_UPDATE_WRITE_READ_BENIGN_FLAG_" + v.Name;

            Procedure result = GetRaceCheckingProcedure(v.tok, updateBenignFlagProcedureName);

            result.InParams = inParams;

            GPUVerifier.AddInlineAttribute(result);

            return result;
        }

        protected Procedure MakeCheckAccessProcedureHeader(Variable v, AccessType access)
        {
            List<Variable> inParams = new List<Variable>();

            Variable predicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);
            Debug.Assert(!(mt.Result is MapType));

            Variable offsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
            Variable valueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

            inParams.Add(predicateParameter);
            inParams.Add(offsetParameter);
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
            {
                inParams.Add(valueParameter);
            }

            string checkProcedureName = "_CHECK_" + access + "_" + v.Name;

            Procedure result = GetRaceCheckingProcedure(v.tok, checkProcedureName);

            result.InParams = inParams;

            return result;
        }

        public void AddRaceCheckingCandidateRequires(Procedure proc)
        {
            foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false))
            {
                AddNoAccessCandidateRequires(proc, v);
                AddReadOrWrittenOffsetIsThreadIdCandidateRequires(proc, v);
            }
        }

        public void AddRaceCheckingCandidateEnsures(Procedure proc)
        {
            foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false))
            {
                AddNoAccessCandidateEnsures(proc, v);
                AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(proc, v);
            }
        }

        private void AddNoAccessCandidateRequires(Procedure proc, Variable v, AccessType access)
        {
            verifier.AddCandidateRequires(proc, NoAccessHasOccurredExpr(v, access));
        }

        private void AddNoAccessCandidateEnsures(Procedure proc, Variable v, AccessType access)
        {
            verifier.AddCandidateEnsures(proc, NoAccessHasOccurredExpr(v, access));
        }

        private HashSet<Expr> GetOffsetsAccessed(IRegion region, Variable v, AccessType access)
        {
            HashSet<Expr> result = new HashSet<Expr>();

            foreach (Cmd c in region.Cmds())
            {
                if (c is CallCmd)
                {
                    CallCmd call = c as CallCmd;

                    if (call.callee == "_LOG_" + access + "_" + v.Name)
                    {
                        // Ins[0] is thread 1's predicate,
                        // Ins[1] is the offset to be read
                        // If Ins[1] has the form BV32_ADD(offset#construct...(P), offset),
                        // we are looking for the second parameter to this BV32_ADD
                        Expr offset = call.Ins[1];
                        if (offset is NAryExpr)
                        {
                            var nExpr = (NAryExpr)offset;
                            if (nExpr.Fun.FunctionName == "BV32_ADD" &&
                                nExpr.Args[0] is NAryExpr)
                            {
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

        public virtual void AddRaceCheckingDeclarations()
        {
            foreach (Variable v in verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false))
                AddRaceCheckingDecsAndProcsForVar(v);
        }

        protected void AddUpdateBenignFlagProcedure(Variable v)
        {
            Procedure updateBenignFlagProcedure = MakeUpdateBenignFlagProcedureHeader(v);

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);

            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE);
            Variable accessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, AccessType.WRITE, verifier.IntRep.GetIntType(verifier.size_t_bits));
            Variable accessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

            Variable predicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
            Variable offsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));

            Debug.Assert(!(mt.Result is MapType));

            List<Variable> locals = new List<Variable>();
            List<BigBlock> bigblocks = new List<BigBlock>();
            List<Cmd> simpleCmds = new List<Cmd>();

            Expr condition = Expr.And(new IdentifierExpr(v.tok, predicateParameter),
                               Expr.And(new IdentifierExpr(v.tok, accessHasOccurredVariable),
                                 Expr.Eq(new IdentifierExpr(v.tok, accessOffsetVariable),
                                   new IdentifierExpr(v.tok, offsetParameter))));

            simpleCmds.Add(MakeConditionalAssignment(accessBenignFlagVariable,
                condition, Expr.False));

            bigblocks.Add(new BigBlock(v.tok, "_UPDATE_BENIGN_FLAG", simpleCmds, null, null));

            Implementation updateBenignFlagImplementation = new Implementation(v.tok, "_UPDATE_WRITE_READ_BENIGN_FLAG_" + v.Name, new List<TypeVariable>(), updateBenignFlagProcedure.InParams, new List<Variable>(), locals, new StmtList(bigblocks, v.tok));
            GPUVerifier.AddInlineAttribute(updateBenignFlagImplementation);

            updateBenignFlagImplementation.Proc = updateBenignFlagProcedure;

            verifier.Program.AddTopLevelDeclaration(updateBenignFlagProcedure);
            verifier.Program.AddTopLevelDeclaration(updateBenignFlagImplementation);
        }

        protected void AddCheckAccessProcedure(Variable v, AccessType access)
        {
            Procedure checkAccessProcedure = MakeCheckAccessProcedureHeader(v, access);

            Variable predicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);
            Debug.Assert(!(mt.Result is MapType));

            Variable offsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));

            if (access == AccessType.READ)
            {
                Variable writeReadBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

                Expr noBenignTest = null;

                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
                {
                    noBenignTest = new IdentifierExpr(Token.NoToken, writeReadBenignFlagVariable);
                }

                AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, noBenignTest, AccessType.WRITE, "write_read");

                if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead)
                {
                    AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, null, AccessType.ATOMIC, "atomic_read");
                }
            }
            else if (access == AccessType.WRITE)
            {
                Variable valueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

                Expr writeNoBenignTest = null;

                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
                {
                    writeNoBenignTest = Expr.Neq(
                        new IdentifierExpr(Token.NoToken, RaceInstrumentationUtil.MakeValueVariable(v.Name, AccessType.WRITE, mt.Result)),
                        new IdentifierExpr(Token.NoToken, valueParameter));
                }

                AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, writeNoBenignTest, AccessType.WRITE, "write_write");

                Expr readNoBenignTest = null;

                if (!GPUVerifyVCGenCommandLineOptions.NoBenign)
                {
                    readNoBenignTest = Expr.Neq(
                        new IdentifierExpr(Token.NoToken, RaceInstrumentationUtil.MakeValueVariable(v.Name, AccessType.READ, mt.Result)),
                        new IdentifierExpr(Token.NoToken, valueParameter));
                }

                AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, readNoBenignTest, AccessType.READ, "read_write");

                if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite)
                {
                    AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, null, AccessType.ATOMIC, "atomic_write");
                }
            }
            else if (access == AccessType.ATOMIC)
            {
                if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite)
                {
                    AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, null, AccessType.WRITE, "write_atomic");
                }

                if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead)
                {
                    AddCheckAccessCheck(v, checkAccessProcedure, predicateParameter, offsetParameter, null, AccessType.READ, "read_atomic");
                }
            }

            verifier.Program.AddTopLevelDeclaration(checkAccessProcedure);
        }

        protected void AddCheckAccessCheck(Variable v, Procedure checkAccessProcedure, Variable predicateParameter, Variable offsetParameter, Expr noBenignTest, AccessType access, string attribute)
        {
            // Check atomic by thread 2 does not conflict with read by thread 1
            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
            Variable accessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits));

            Expr accessGuard = new IdentifierExpr(Token.NoToken, predicateParameter);
            accessGuard = Expr.And(accessGuard, new IdentifierExpr(Token.NoToken, accessHasOccurredVariable));
            accessGuard = Expr.And(accessGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, accessOffsetVariable),
                                      new IdentifierExpr(Token.NoToken, offsetParameter)));

            if (noBenignTest != null)
                accessGuard = Expr.And(accessGuard, noBenignTest);

            if (verifier.KernelArrayInfo.GetGroupSharedArrays(false).Contains(v))
                accessGuard = Expr.And(accessGuard, GPUVerifier.ThreadsInSameGroup());

            accessGuard = Expr.Not(accessGuard);

            Requires noAccessRaceRequires = new Requires(false, accessGuard);

            string sourceName = verifier.GlobalArraySourceNames[v.Name];
            Debug.Assert(sourceName != null);

            noAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, attribute, new List<object>(), null);
            noAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), noAccessRaceRequires.Attributes);
            noAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, noAccessRaceRequires.Attributes);
            noAccessRaceRequires.Attributes = new QKeyValue(Token.NoToken, "source_name", new List<object>() { sourceName }, noAccessRaceRequires.Attributes);
            checkAccessProcedure.Requires.Add(noAccessRaceRequires);
        }

        protected void AddLogRaceDeclarations(Variable v, AccessType access)
        {
            verifier.FindOrCreateAccessHasOccurredVariable(v.Name, access);
            verifier.FindOrCreateOffsetVariable(v.Name, access);

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
            {
                Debug.Assert(v.TypedIdent.Type is MapType);
                MapType mt = v.TypedIdent.Type as MapType;
                Debug.Assert(mt.Arguments.Count == 1);
                verifier.FindOrCreateValueVariable(v.Name, access, mt.Result);
            }

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
            {
                Debug.Assert(v.TypedIdent.Type is MapType);
                MapType mt = v.TypedIdent.Type as MapType;
                Debug.Assert(mt.Arguments.Count == 1);
                verifier.FindOrCreateBenignFlagVariable(v.Name);
            }

            if ((access == AccessType.READ || access == AccessType.WRITE) &&
                 verifier.ArraysAccessedByAsyncWorkGroupCopy[access].Contains(v.Name))
            {
                verifier.FindOrCreateAsyncHandleVariable(v.Name, access);
            }
        }

        protected static AssignCmd MakeConditionalAssignment(Variable lhs, Expr condition, Expr rhs)
        {
            List<AssignLhs> lhss = new List<AssignLhs>();
            List<Expr> rhss = new List<Expr>();
            lhss.Add(new SimpleAssignLhs(lhs.tok, new IdentifierExpr(lhs.tok, lhs)));
            rhss.Add(new NAryExpr(rhs.tok, new IfThenElse(rhs.tok), new List<Expr>(new Expr[] { condition, rhs, new IdentifierExpr(lhs.tok, lhs) })));
            return new AssignCmd(lhs.tok, lhss, rhss);
        }

        private Expr MakeAccessedIndex(Variable v, Expr offsetExpr, AccessType access)
        {
            Expr result = new IdentifierExpr(v.tok, v.Clone() as Variable);
            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);

            result = Expr.Select(result,
                new Expr[] { offsetExpr });
            Debug.Assert(!(mt.Result is MapType));
            return result;
        }

        protected void AddRequiresNoPendingAccess(Variable v)
        {
            IdentifierExpr readAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.READ));
            IdentifierExpr writeAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.WRITE));
            IdentifierExpr atomicAccessOccurred1 = new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, AccessType.ATOMIC));

            foreach (var proc in verifier.KernelProcedures.Keys)
                proc.Requires.Add(new Requires(false, Expr.And(Expr.And(Expr.Not(readAccessOccurred1), Expr.Not(writeAccessOccurred1)), Expr.Not(atomicAccessOccurred1))));
        }

        private Expr BuildAccessOccurredFalseExpr(string name, AccessType access)
        {
            return Expr.Imp(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(name, access)),
                                               Expr.False);
        }

        private AssertCmd BuildAccessOccurredFalseInvariant(string name, AccessType access)
        {
            return new AssertCmd(Token.NoToken, BuildAccessOccurredFalseExpr(name, access));
        }

        private Expr AccessHasOccurredExpr(Variable v, AccessType access)
        {
            return new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access));
        }

        private Expr NoAccessHasOccurredExpr(Variable v, AccessType access)
        {
            return Expr.Not(AccessHasOccurredExpr(v, access));
        }

        protected void AddOffsetsSatisfyPredicatesCandidateInvariant(IRegion region, Variable v, AccessType access, List<Expr> preds)
        {
            if (preds.Count != 0)
            {
                Expr expr = AccessedOffsetsSatisfyPredicatesExpr(v, preds, access);
                verifier.AddCandidateInvariant(region, expr, "accessedOffsetsSatisfyPredicates");
            }
        }

        private Expr AccessedOffsetsSatisfyPredicatesExpr(Variable v, IEnumerable<Expr> offsets, AccessType access)
        {
            return Expr.Imp(
                    new IdentifierExpr(Token.NoToken, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)),
                    offsets.Aggregate(Expr.Or));
        }

        private Expr AccessedOffsetIsThreadLocalIdExpr(Variable v, AccessType access)
        {
            Expr offsetVar = new IdentifierExpr(v.tok, RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits)));
            Expr offsetExpr = verifier.IntRep.MakeZext(new IdentifierExpr(v.tok, GPUVerifier.MakeThreadId("X", 1)), offsetVar.Type);
            return Expr.Imp(
                      new IdentifierExpr(v.tok, GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access)),
                      Expr.Eq(offsetVar, offsetExpr));
        }

        private Expr GlobalIdExpr(string dimension, int thread)
        {
            return new VariableDualiser(thread, null, null).VisitExpr(verifier.GlobalIdExpr(dimension).Clone() as Expr);
        }

        protected void AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(IRegion region, Variable v, Expr constant, AccessType access)
        {
            Expr expr = MakeCTimesLocalIdRangeExpression(v, constant, access);
            verifier.AddCandidateInvariant(region,
                expr, "accessedOffsetInRangeCTimesLid");
        }

        private Expr MakeCTimesLocalIdRangeExpression(Variable v, Expr constant, AccessType access)
        {
            Expr cTimesLocalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
                new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", 1)));

            Expr cTimesLocalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
                new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", 1))), constant.Clone() as Expr);

            Expr cTimesLocalIdLeqAccessedOffset = verifier.IntRep.MakeSle(cTimesLocalId, OffsetXExpr(v, access, 1));

            Expr accessedOffsetLtCTimesLocalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, access, 1), cTimesLocalIdPlusC);

            return Expr.Imp(
                    GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access),
                    Expr.And(cTimesLocalIdLeqAccessedOffset, accessedOffsetLtCTimesLocalIdPlusC));
        }

        private IdentifierExpr OffsetXExpr(Variable v, AccessType access, int thread)
        {
            return new IdentifierExpr(v.tok, new VariableDualiser(thread, null, null).VisitVariable(RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits))));
        }

        protected void AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(IRegion region, Variable v, Expr constant, AccessType access)
        {
            Expr expr = MakeCTimesGlobalIdRangeExpr(v, constant, access);
            verifier.AddCandidateInvariant(region,
                expr, "accessedOffsetInRangeCTimesGid");
        }

        private Expr MakeCTimesGlobalIdRangeExpr(Variable v, Expr constant, AccessType access)
        {
            Expr cTimesGlobalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
                GlobalIdExpr("X", 1));

            Expr cTimesGlobalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
                GlobalIdExpr("X", 1)), constant.Clone() as Expr);

            Expr cTimesGlobalIdLeqAccessedOffset = verifier.IntRep.MakeSle(cTimesGlobalId, OffsetXExpr(v, access, 1));

            Expr accessedOffsetLtCTimesGlobalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, access, 1), cTimesGlobalIdPlusC);

            Expr implication = Expr.Imp(
                    GPUVerifier.MakeAccessHasOccurredExpr(v.Name, access),
                    Expr.And(cTimesGlobalIdLeqAccessedOffset, accessedOffsetLtCTimesGlobalIdPlusC));
            return implication;
        }

        protected void AddAccessedOffsetIsThreadLocalIdCandidateRequires(Procedure proc, Variable v, AccessType access)
        {
            verifier.AddCandidateRequires(proc, AccessedOffsetIsThreadLocalIdExpr(v, access));
        }

        protected void AddAccessedOffsetIsThreadLocalIdCandidateEnsures(Procedure proc, Variable v, AccessType access)
        {
            verifier.AddCandidateEnsures(proc, AccessedOffsetIsThreadLocalIdExpr(v, access));
        }
    }

    class FindReferencesToNamedVariableVisitor : StandardVisitor
    {
        internal bool found = false;
        private string name;

        internal FindReferencesToNamedVariableVisitor(string name)
        {
            this.name = name;
        }

        public override Variable VisitVariable(Variable node)
        {
            if (GVUtil.StripThreadIdentifier(node.Name).Equals(name))
            {
                found = true;
            }

            return base.VisitVariable(node);
        }
    }

    class ImplementationInstrumenter
    {
        private RaceInstrumenter ri;
        private GPUVerifier verifier;
        private Implementation impl;
        private QKeyValue sourceLocationAttributes = null;

        internal ImplementationInstrumenter(RaceInstrumenter ri, Implementation impl)
        {
            this.ri = ri;
            this.verifier = ri.verifier;
            this.impl = impl;
        }

        internal void AddRaceCheckCalls()
        {
            impl.Blocks = impl.Blocks.Select(AddRaceCheckCalls).ToList();
        }

        private void AddRaceCheckCalls(Implementation impl)
        {
        }

        private Block AddRaceCheckCalls(Block b)
        {
            b.Cmds = AddRaceCheckCalls(b.Cmds);
            return b;
        }

        private List<Cmd> AddRaceCheckCalls(List<Cmd> cs)
        {
            var result = new List<Cmd>();
            foreach (Cmd c in cs)
            {
                if (c is AssertCmd)
                {
                    AssertCmd assertion = c as AssertCmd;
                    if (QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc"))
                    {
                        sourceLocationAttributes = assertion.Attributes;
                        // Remove source location assertions
                        continue;
                    }
                }

                if (c is CallCmd)
                {
                    CallCmd call = c as CallCmd;
                    if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic"))
                    {
                        AddLogAndCheckCalls(result, new AccessRecord((call.Ins[0] as IdentifierExpr).Decl, call.Ins[1]), AccessType.ATOMIC, null);
                        Debug.Assert(call.Outs.Count() == 2); // The receiving variable and the array should be assigned to
                        result.Add(new HavocCmd(Token.NoToken, new List<IdentifierExpr> { call.Outs[0] })); // We havoc the receiving variable.  We do not need to havoc the array, because it *must* be the case that this array is modelled adversarially
                        continue;
                    }

                    if (QKeyValue.FindBoolAttribute(call.Attributes, "async_work_group_copy"))
                    {
                        Procedure asyncWorkGroupCopy = FindOrCreateAsyncWorkGroupCopy(((IdentifierExpr)call.Outs[1]).Decl, ((IdentifierExpr)call.Ins[1]).Decl);
                        CallCmd asyncWorkGroupCopyCall = new CallCmd(Token.NoToken, asyncWorkGroupCopy.Name,
                          new List<Expr> { call.Ins[0], call.Ins[2], call.Ins[3], call.Ins[4] },
                          new List<IdentifierExpr> { call.Outs[0] });
                        asyncWorkGroupCopyCall.Proc = asyncWorkGroupCopy;
                        result.Add(asyncWorkGroupCopyCall);
                        continue;
                    }

                    if (QKeyValue.FindBoolAttribute(call.Attributes, "wait_group_events"))
                    {
                        Expr handle = call.Ins[0];
                        var sourceLocAttributes = new QKeyValue(Token.NoToken, "sourceloc_num", new List<object> { QKeyValue.FindExprAttribute(call.Attributes, "sourceloc_num") }, null);

                        // Assert that the threads are uniformly enabled
                        result.Add(new AssertCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(Expr.Ident(verifier.FindOrCreateEnabledVariable())), sourceLocAttributes));

                        // Assert that the handle passed is uniform
                        result.Add(new AssertCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(handle), sourceLocAttributes));

                        foreach (var access in verifier.ArraysAccessedByAsyncWorkGroupCopy.Keys)
                        {
                            foreach (var array in verifier.ArraysAccessedByAsyncWorkGroupCopy[access])
                            {
                                // Set the handle associated with an array access to the "no handle"
                                // value if it's current handle matches the given handle
                                IdentifierExpr handleVariable = Expr.Ident(RaceInstrumentationUtil.MakeAsyncHandleVariable(array, access, verifier.IntRep.GetIntType(verifier.size_t_bits)));
                                result.Add(new AssignCmd(Token.NoToken,
                                  new List<AssignLhs> { new SimpleAssignLhs(Token.NoToken, handleVariable) },
                                  new List<Expr> { new NAryExpr(Token.NoToken, new IfThenElse(Token.NoToken),
                                  new List<Expr> {
                                    Expr.Eq(handle, handleVariable),
                                    verifier.FindOrCreateAsyncNoHandleConstant(),
                                    handleVariable }) }));
                            }
                        }

                        continue;
                    }
                }

                if (c is AssignCmd)
                {
                    AssignCmd assign = c as AssignCmd;

                    ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
                    foreach (var rhs in assign.Rhss)
                        rc.Visit(rhs);
                    if (rc.nonPrivateAccesses.Count > 0)
                    {
                        foreach (AccessRecord ar in rc.nonPrivateAccesses)
                        {
                            // Ignore disabled arrays
                            if (verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(ar.v))
                            {
                                // Ignore read-only arrays (whether or not they are disabled)
                                if (!verifier.KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(ar.v))
                                {
                                    AddLogAndCheckCalls(result, ar, AccessType.READ, null);
                                }
                            }
                        }
                    }

                    foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss))
                    {
                        WriteCollector wc = new WriteCollector(verifier.KernelArrayInfo);
                        wc.Visit(LhsRhs.Item1);
                        if (wc.FoundNonPrivateWrite())
                        {
                            // Ignore disabled arrays
                            if (verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(wc.GetAccess().v))
                            {
                                AddLogAndCheckCalls(result, wc.GetAccess(), AccessType.WRITE, LhsRhs.Item2);
                            }
                        }
                    }
                }

                result.Add(c);
            }

            return result;
        }

        private Procedure FindOrCreateAsyncWorkGroupCopy(Variable dstArray, Variable srcArray)
        {
            string procedureName = "_ASYNC_WORK_GROUP_COPY_" + dstArray.Name + "_" + srcArray.Name;

            List<Procedure> candidateProcedures =
              verifier.Program.TopLevelDeclarations.OfType<Procedure>().Where(
                item => item.Name == procedureName).ToList();
            if (candidateProcedures.Count() > 0)
            {
                Debug.Assert(candidateProcedures.Count() == 1);
                return candidateProcedures[0];
            }

            IdentifierExpr dstOffset = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "DstOffset", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            IdentifierExpr srcOffset = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "SrcOffset", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            IdentifierExpr size = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "Size", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            IdentifierExpr handle = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "Handle", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            List<Variable> inParams = new List<Variable> { dstOffset.Decl, srcOffset.Decl, size.Decl, handle.Decl };

            IdentifierExpr resultHandle = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "ResultHandle", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            List<Variable> outParams = new List<Variable> { resultHandle.Decl };

            IdentifierExpr index = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "Index", verifier.IntRep.GetIntType(verifier.size_t_bits))));
            IdentifierExpr idX = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "IdX", GPUVerifier._X.TypedIdent.Type)));
            IdentifierExpr idY = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "IdY", GPUVerifier._Y.TypedIdent.Type)));
            IdentifierExpr idZ = Expr.Ident(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "IdZ", GPUVerifier._Z.TypedIdent.Type)));
            List<Variable> locals = new List<Variable> { index.Decl, idX.Decl, idY.Decl, idZ.Decl };

            List<Requires> preconditions = new List<Requires>() {
                RequireEqualBetweenThreadsInSameGroup(Expr.Ident(verifier.FindOrCreateEnabledVariable())),
                RequireEqualBetweenThreadsInSameGroup(dstOffset),
                RequireEqualBetweenThreadsInSameGroup(srcOffset),
                RequireEqualBetweenThreadsInSameGroup(size),
                RequireEqualBetweenThreadsInSameGroup(handle)
            };

            Block entryBlock = new Block(Token.NoToken, "entry", new List<Cmd>(), null);
            Block accessBlock = new Block(Token.NoToken, "access", new List<Cmd>(), null);
            Block noAccessBlock = new Block(Token.NoToken, "no_access", new List<Cmd>(), null);
            Block exitBlock = new Block(Token.NoToken, "exit", new List<Cmd>(), null);
            List<Block> blocks = new List<Block> { entryBlock, accessBlock, exitBlock };

            entryBlock.TransferCmd = new GotoCmd(Token.NoToken,
              new List<string> { "access", "no_access" },
              new List<Block> { accessBlock, noAccessBlock });
            accessBlock.TransferCmd = new GotoCmd(Token.NoToken,
              new List<string> { "exit" },
              new List<Block> { exitBlock });
            noAccessBlock.TransferCmd = new GotoCmd(Token.NoToken,
              new List<string> { "exit" },
              new List<Block> { exitBlock });
            exitBlock.TransferCmd = new ReturnCmd(Token.NoToken);

            entryBlock.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.Neq(resultHandle, verifier.FindOrCreateAsyncNoHandleConstant())));
            entryBlock.Cmds.Add(new AssignCmd(Token.NoToken, new List<AssignLhs> { new SimpleAssignLhs(Token.NoToken, resultHandle) },
                    new List<Expr> { new NAryExpr(Token.NoToken, new IfThenElse(Token.NoToken),
                    new List<Expr> {
                        Expr.Eq(handle, verifier.FindOrCreateAsyncNoHandleConstant()), resultHandle, handle
                    }) }));
            entryBlock.Cmds.Add(new AssumeCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(resultHandle)));

            // Choose arbitrary numbers X, Y and Z that are equal between the threads.
            entryBlock.Cmds.Add(new AssumeCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(idX)));
            entryBlock.Cmds.Add(new AssumeCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(idY)));
            entryBlock.Cmds.Add(new AssumeCmd(Token.NoToken, EqualBetweenThreadsInSameGroup(idZ)));

            // If X, Y and Z match the thread's local id, issue a read and write by this thread relative
            // to an arbitrarily chosen index in the range specified by the async copy
            Expr idsMatch = Expr.And(Expr.Eq(idX, Expr.Ident(GPUVerifier._X)),
                            Expr.And(Expr.Eq(idY, Expr.Ident(GPUVerifier._Y)),
                                     Expr.Eq(idZ, Expr.Ident(GPUVerifier._Z))));
            Expr accessedValue = Expr.Select(Expr.Ident(srcArray), new Expr[] { verifier.IntRep.MakeAdd(srcOffset, index) });
            accessedValue.Type = (srcArray.TypedIdent.Type as MapType).Result;
            accessBlock.Cmds.Add(new AssumeCmd(Token.NoToken, idsMatch, new QKeyValue(Token.NoToken, "partition", new List<object>(), null)));
            accessBlock.Cmds.Add(new AssumeCmd(Token.NoToken, verifier.IntRep.MakeUge(index, verifier.Zero(verifier.size_t_bits))));
            accessBlock.Cmds.Add(new AssumeCmd(Token.NoToken, verifier.IntRep.MakeUlt(index, size)));
            accessBlock.Cmds.Add(MakeLogCall(new AccessRecord(dstArray, verifier.IntRep.MakeAdd(dstOffset, index)), AccessType.WRITE, accessedValue, resultHandle));
            accessBlock.Cmds.Add(MakeCheckCall(accessBlock.Cmds, new AccessRecord(dstArray, verifier.IntRep.MakeAdd(dstOffset, index)), AccessType.WRITE, accessedValue));
            accessBlock.Cmds.Add(MakeLogCall(new AccessRecord(srcArray, verifier.IntRep.MakeAdd(srcOffset, index)), AccessType.READ, accessedValue, resultHandle));
            accessBlock.Cmds.Add(MakeCheckCall(accessBlock.Cmds, new AccessRecord(srcArray, verifier.IntRep.MakeAdd(srcOffset, index)), AccessType.READ, accessedValue));

            noAccessBlock.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.Not(idsMatch), new QKeyValue(Token.NoToken, "partition", new List<object>(), null)));

            Procedure asyncWorkGroupCopyProcedure = new Procedure(Token.NoToken, procedureName, new List<TypeVariable>(),
              inParams, outParams, preconditions,
              new List<IdentifierExpr>(), new List<Ensures>());
            GPUVerifier.AddInlineAttribute(asyncWorkGroupCopyProcedure);

            Implementation asyncWorkGroupCopyImplementation = new Implementation(Token.NoToken,
              procedureName, new List<TypeVariable>(), inParams, outParams, locals, blocks);
            asyncWorkGroupCopyImplementation.Proc = asyncWorkGroupCopyProcedure;
            GPUVerifier.AddInlineAttribute(asyncWorkGroupCopyImplementation);

            verifier.Program.AddTopLevelDeclaration(asyncWorkGroupCopyProcedure);
            verifier.Program.AddTopLevelDeclaration(asyncWorkGroupCopyImplementation);

            return asyncWorkGroupCopyProcedure;
        }

        private Requires RequireEqualBetweenThreadsInSameGroup(Expr e)
        {
            Requires result = new Requires(false, EqualBetweenThreadsInSameGroup(e));
            result.Attributes = new QKeyValue(Token.NoToken, "sourceloc_num", new List<object> { QKeyValue.FindExprAttribute(sourceLocationAttributes, "sourceloc_num") }, null);
            return result;
        }

        private Expr EqualBetweenThreadsInSameGroup(Expr e)
        {
            return Expr.Imp(GPUVerifier.ThreadsInSameGroup(), Expr.Eq(e,
                    new NAryExpr(Token.NoToken, new FunctionCall(verifier.FindOrCreateOther(verifier.size_t_bits)), new List<Expr> { e })));
        }

        private void AddLogAndCheckCalls(List<Cmd> result, AccessRecord ar, AccessType access, Expr value)
        {
            // We should not be adding these calls for any disabled arrays
            Debug.Assert(verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(ar.v));

            // We also should not be adding these calls for any read-only arrays (whether or not they are disabled)
            Debug.Assert(!verifier.KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(ar.v));

            result.Add(MakeLogCall(ar, access, value, null));
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
                result.Add(MakeUpdateBenignFlagCall(ar));

            if (!GPUVerifyVCGenCommandLineOptions.OnlyLog)
                result.Add(MakeCheckCall(result, ar, access, value));
        }

        private CallCmd MakeCheckCall(List<Cmd> result, AccessRecord ar, AccessType access, Expr value)
        {
            if (sourceLocationAttributes == null)
                ExitWithNoSourceError(ar.v, access);

            List<Expr> inParamsChk = new List<Expr>();
            inParamsChk.Add(ar.Index);
            MaybeAddValueParameter(inParamsChk, ar, value, access);
            Procedure checkProcedure = ri.GetRaceCheckingProcedure(Token.NoToken, "_CHECK_" + access + "_" + ar.v.Name);
            verifier.OnlyThread2.Add(checkProcedure.Name);
            string checkState = "check_state_" + ri.CheckStateCounter;
            ri.CheckStateCounter++;
            AssumeCmd captureStateAssume = new AssumeCmd(Token.NoToken, Expr.True);
            captureStateAssume.Attributes = sourceLocationAttributes.Clone() as QKeyValue;
            captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
              "captureState", new List<object>() { checkState }, captureStateAssume.Attributes);
            captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
              "check_id", new List<object>() { checkState }, captureStateAssume.Attributes);
            captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
              "do_not_predicate", new List<object>() { }, captureStateAssume.Attributes);

            result.Add(captureStateAssume);
            CallCmd checkAccessCallCmd = new CallCmd(Token.NoToken, checkProcedure.Name, inParamsChk, new List<IdentifierExpr>());
            checkAccessCallCmd.Proc = checkProcedure;
            checkAccessCallCmd.Attributes = sourceLocationAttributes.Clone() as QKeyValue;
            checkAccessCallCmd.Attributes = new QKeyValue(Token.NoToken, "check_id", new List<object>() { checkState }, checkAccessCallCmd.Attributes);
            return checkAccessCallCmd;
        }

        private CallCmd MakeLogCall(AccessRecord ar, AccessType access, Expr value, Expr asyncHandle)
        {
            if (sourceLocationAttributes == null)
                ExitWithNoSourceError(ar.v, access);

            List<Expr> inParamsLog = new List<Expr>();
            inParamsLog.Add(ar.Index);
            MaybeAddValueParameter(inParamsLog, ar, value, access);
            MaybeAddValueOldParameter(inParamsLog, ar, access);
            MaybeAddAsyncHandleParameter(inParamsLog, ar, asyncHandle, access);
            Procedure logProcedure = ri.GetRaceCheckingProcedure(Token.NoToken, "_LOG_" + access + "_" + ar.v.Name);
            verifier.OnlyThread1.Add(logProcedure.Name);
            CallCmd logAccessCallCmd = new CallCmd(Token.NoToken, logProcedure.Name, inParamsLog, new List<IdentifierExpr>());
            logAccessCallCmd.Proc = logProcedure;
            logAccessCallCmd.Attributes = sourceLocationAttributes.Clone() as QKeyValue;
            return logAccessCallCmd;
        }

        private void ExitWithNoSourceError(Variable v, AccessType access)
        {
            Console.Error.WriteLine("No source location information available when processing " +
              access + " operation on " + v + " at " + GPUVerifyVCGenCommandLineOptions.inputFiles[0] + ":" +
              v.tok.line + ":" + v.tok.col + ".  Aborting.");
            Environment.Exit(1);
        }

        private void MaybeAddValueParameter(List<Expr> parameters, AccessRecord ar, Expr value, AccessType access)
        {
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
            {
                if (value != null)
                {
                    parameters.Add(value);
                }
                else
                {
                    // TODO: Why do we do this?  Seems wrong to assume that if no Value is supplied
                    // then the value being written is the same as the value that was already there
                    Expr e = Expr.Select(new IdentifierExpr(Token.NoToken, ar.v), new Expr[] { ar.Index });
                    e.Type = (ar.v.TypedIdent.Type as MapType).Result;
                    parameters.Add(e);
                }
            }
        }

        private void MaybeAddValueOldParameter(List<Expr> parameters, AccessRecord ar, AccessType access)
        {
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
            {
                Expr e = Expr.Select(new IdentifierExpr(Token.NoToken, ar.v), new Expr[] { ar.Index });
                e.Type = (ar.v.TypedIdent.Type as MapType).Result;
                parameters.Add(e);
            }
        }

        private void MaybeAddAsyncHandleParameter(List<Expr> parameters, AccessRecord ar, Expr asyncHandle, AccessType access)
        {
            if (!(new AccessType[] { AccessType.READ, AccessType.WRITE }).Contains(access))
                return;

            if (verifier.ArraysAccessedByAsyncWorkGroupCopy[access].Contains(ar.v.Name))
            {
                if (asyncHandle != null)
                    parameters.Add(asyncHandle);
                else
                    parameters.Add(verifier.FindOrCreateAsyncNoHandleConstant());
            }
            else
            {
                Debug.Assert(asyncHandle == null);
            }
        }

        private CallCmd MakeUpdateBenignFlagCall(AccessRecord ar)
        {
            List<Expr> inParamsUpdateBenignFlag = new List<Expr>();
            inParamsUpdateBenignFlag.Add(ar.Index);
            Procedure updateBenignFlagProcedure = ri.GetRaceCheckingProcedure(Token.NoToken, "_UPDATE_WRITE_READ_BENIGN_FLAG_" + ar.v.Name);
            verifier.OnlyThread2.Add(updateBenignFlagProcedure.Name);
            CallCmd updateBenignFlagCallCmd = new CallCmd(Token.NoToken, updateBenignFlagProcedure.Name, inParamsUpdateBenignFlag, new List<IdentifierExpr>());
            updateBenignFlagCallCmd.Proc = updateBenignFlagProcedure;
            return updateBenignFlagCallCmd;
        }
    }
}
