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
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.Basetypes;
    using Microsoft.Boogie;

    public class KernelDualiser
    {
        public GPUVerifier Verifier { get; }

        private List<BarrierInvariantDescriptor> barrierInvariantDescriptors = new List<BarrierInvariantDescriptor>();

        public KernelDualiser(GPUVerifier verifier)
        {
            Verifier = verifier;
        }

        private string procName = null;

        private void DualiseProcedure(Procedure proc)
        {
            procName = proc.Name;

            proc.Requires = DualiseRequires(proc.Requires);
            proc.Ensures = DualiseEnsures(proc.Ensures);
            proc.Modifies = DualiseModifies(proc.Modifies);

            proc.InParams = DualiseVariableSequence(proc.InParams);
            proc.OutParams = DualiseVariableSequence(proc.OutParams);

            procName = null;
        }

        private List<Requires> DualiseRequires(List<Requires> requiresSeq)
        {
            List<Requires> newRequires = new List<Requires>();
            foreach (Requires r in requiresSeq)
            {
                newRequires.Add(MakeThreadSpecificRequires(r, 1));
                if (!ContainsAsymmetricExpression(r.Condition)
                    && !Verifier.UniformityAnalyser.IsUniform(procName, r.Condition))
                {
                    newRequires.Add(MakeThreadSpecificRequires(r, 2));
                }
            }

            return newRequires;
        }

        private List<Ensures> DualiseEnsures(List<Ensures> ensuresSeq)
        {
            List<Ensures> newEnsures = new List<Ensures>();
            foreach (Ensures e in ensuresSeq)
            {
                newEnsures.Add(MakeThreadSpecificEnsures(e, 1));
                if (!ContainsAsymmetricExpression(e.Condition)
                    && !Verifier.UniformityAnalyser.IsUniform(procName, e.Condition))
                {
                    newEnsures.Add(MakeThreadSpecificEnsures(e, 2));
                }
            }

            return newEnsures;
        }

        private List<IdentifierExpr> DualiseModifies(List<IdentifierExpr> modifiesSeq)
        {
            List<IdentifierExpr> newModifies = new List<IdentifierExpr>();
            foreach (var m in modifiesSeq)
            {
                newModifies.Add((IdentifierExpr)MakeThreadSpecificExpr(m, 1));
                if (!ContainsAsymmetricExpression(m)
                    && !Verifier.UniformityAnalyser.IsUniform(procName, m))
                {
                    newModifies.Add((IdentifierExpr)MakeThreadSpecificExpr(m, 2));
                }
            }

            return newModifies;
        }

        private Expr MakeThreadSpecificExpr(Expr e, int thread)
        {
            return new VariableDualiser(thread, Verifier, procName).
              VisitExpr(e.Clone() as Expr);
        }

        private Requires MakeThreadSpecificRequires(Requires r, int thread)
        {
            Requires newR = new Requires(r.Free, MakeThreadSpecificExpr(r.Condition, thread));
            newR.Attributes = MakeThreadSpecificAttributes(r.Attributes, thread);
            return newR;
        }

        private Ensures MakeThreadSpecificEnsures(Ensures e, int thread)
        {
            Ensures newE = new Ensures(e.Free, MakeThreadSpecificExpr(e.Condition, thread));
            newE.Attributes = MakeThreadSpecificAttributes(e.Attributes, thread);
            return newE;
        }

        private AssertCmd MakeThreadSpecificAssert(AssertCmd a, int thread)
        {
            AssertCmd result = new AssertCmd(
                Token.NoToken,
                new VariableDualiser(thread, Verifier, procName).VisitExpr(a.Expr.Clone() as Expr),
                MakeThreadSpecificAttributes(a.Attributes, thread));
            return result;
        }

        private AssumeCmd MakeThreadSpecificAssumeFromAssert(AssertCmd a, int thread)
        {
            AssumeCmd result = new AssumeCmd(
                Token.NoToken,
                new VariableDualiser(thread, Verifier, procName).VisitExpr(a.Expr.Clone() as Expr));
            return result;
        }

        public QKeyValue MakeThreadSpecificAttributes(QKeyValue attributes, int thread)
        {
            if (attributes == null)
            {
                return null;
            }

            QKeyValue result = (QKeyValue)attributes.Clone();
            result.AddLast(new QKeyValue(
                Token.NoToken, "thread", new List<object> { new LiteralExpr(Token.NoToken, BigNum.FromInt(thread)) }, null));
            return result;
        }

        private void MakeDual(List<Cmd> cs, Cmd c)
        {
            if (c is CallCmd)
            {
                CallCmd call = c as CallCmd;

                if (QKeyValue.FindBoolAttribute(call.Proc.Attributes, "barrier_invariant"))
                {
                    // There may be a predicate, and there must be an invariant expression and at least one instantiation
                    Debug.Assert(call.Ins.Count >= (2 + (Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1)));
                    var biDescriptor = new UnaryBarrierInvariantDescriptor(
                        Verifier.UniformityAnalyser.IsUniform(call.callee) ? Expr.True : call.Ins[0],
                        Expr.Neq(call.Ins[Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1], Verifier.Zero(1)),
                        call.Attributes,
                        this,
                        procName,
                        Verifier);

                    for (var i = 1 + (Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1); i < call.Ins.Count; i++)
                        biDescriptor.AddInstantiationExpr(call.Ins[i]);

                    barrierInvariantDescriptors.Add(biDescriptor);
                    return;
                }

                if (QKeyValue.FindBoolAttribute(call.Proc.Attributes, "binary_barrier_invariant"))
                {
                    // There may be a predicate, and there must be an invariant expression and at least one pair of
                    // instantiation expressions
                    Debug.Assert(call.Ins.Count >= (3 + (Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1)));
                    var biDescriptor = new BinaryBarrierInvariantDescriptor(
                        Verifier.UniformityAnalyser.IsUniform(call.callee) ? Expr.True : call.Ins[0],
                        Expr.Neq(call.Ins[Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1], Verifier.Zero(1)),
                        call.Attributes,
                        this,
                        procName,
                        Verifier);

                    for (var i = 1 + (Verifier.UniformityAnalyser.IsUniform(call.callee) ? 0 : 1); i < call.Ins.Count; i += 2)
                        biDescriptor.AddInstantiationExprPair(call.Ins[i], call.Ins[i + 1]);

                    barrierInvariantDescriptors.Add(biDescriptor);
                    return;
                }

                if (GPUVerifier.IsBarrier(call.Proc))
                {
                    // Assert barrier invariants
                    foreach (var biIDescriptor in barrierInvariantDescriptors)
                    {
                        QKeyValue sourceLocationInfo = biIDescriptor.GetSourceLocationInfo();
                        cs.Add(biIDescriptor.GetAssertCmd());
                        var vd = new VariableDualiser(1, Verifier, procName);
                        if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
                        {
                            foreach (Expr accessExpr in biIDescriptor.GetAccessedExprs())
                            {
                                var assert = new AssertCmd(Token.NoToken, accessExpr, MakeThreadSpecificAttributes(sourceLocationInfo, 1));
                                assert.Attributes = new QKeyValue(
                                    Token.NoToken, "barrier_invariant_access_check", new List<object> { Expr.True }, assert.Attributes);
                                cs.Add(vd.VisitAssertCmd(assert));
                            }
                        }
                    }
                }

                List<Expr> uniformNewIns = new List<Expr>();
                List<Expr> nonUniformNewIns = new List<Expr>();

                for (int i = 0; i < call.Ins.Count; i++)
                {
                    if (Verifier.UniformityAnalyser.knowsOf(call.callee)
                        && Verifier.UniformityAnalyser.IsUniform(call.callee, Verifier.UniformityAnalyser.GetInParameter(call.callee, i)))
                    {
                        uniformNewIns.Add(call.Ins[i]);
                    }
                    else if (!Verifier.OnlyThread2.Contains(call.callee))
                    {
                        nonUniformNewIns.Add(new VariableDualiser(1, Verifier, procName).VisitExpr(call.Ins[i]));
                    }
                }

                for (int i = 0; i < call.Ins.Count; i++)
                {
                    if (!(Verifier.UniformityAnalyser.knowsOf(call.callee)
                        && Verifier.UniformityAnalyser.IsUniform(call.callee, Verifier.UniformityAnalyser.GetInParameter(call.callee, i)))
                        && !Verifier.OnlyThread1.Contains(call.callee))
                    {
                        nonUniformNewIns.Add(new VariableDualiser(2, Verifier, procName).VisitExpr(call.Ins[i]));
                    }
                }

                List<Expr> newIns = uniformNewIns;
                newIns.AddRange(nonUniformNewIns);

                List<IdentifierExpr> uniformNewOuts = new List<IdentifierExpr>();
                List<IdentifierExpr> nonUniformNewOuts = new List<IdentifierExpr>();
                for (int i = 0; i < call.Outs.Count; i++)
                {
                    if (Verifier.UniformityAnalyser.knowsOf(call.callee)
                        && Verifier.UniformityAnalyser.IsUniform(call.callee, Verifier.UniformityAnalyser.GetOutParameter(call.callee, i)))
                    {
                        uniformNewOuts.Add(call.Outs[i]);
                    }
                    else
                    {
                        nonUniformNewOuts.Add(new VariableDualiser(1, Verifier, procName).VisitIdentifierExpr(call.Outs[i].Clone() as IdentifierExpr) as IdentifierExpr);
                    }
                }

                for (int i = 0; i < call.Outs.Count; i++)
                {
                    if (!(Verifier.UniformityAnalyser.knowsOf(call.callee)
                        && Verifier.UniformityAnalyser.IsUniform(call.callee, Verifier.UniformityAnalyser.GetOutParameter(call.callee, i))))
                    {
                        nonUniformNewOuts.Add(new VariableDualiser(2, Verifier, procName).VisitIdentifierExpr(call.Outs[i].Clone() as IdentifierExpr) as IdentifierExpr);
                    }
                }

                List<IdentifierExpr> newOuts = uniformNewOuts;
                newOuts.AddRange(nonUniformNewOuts);

                CallCmd newCallCmd = new CallCmd(call.tok, call.callee, newIns, newOuts);

                newCallCmd.Proc = call.Proc;

                newCallCmd.Attributes = call.Attributes;

                if (newCallCmd.callee.StartsWith("_LOG_ATOMIC"))
                {
                    QKeyValue curr = newCallCmd.Attributes;
                    if (curr.Key.StartsWith("arg"))
                        newCallCmd.Attributes = new QKeyValue(Token.NoToken, curr.Key, new List<object>(new object[] { Dualise(curr.Params[0] as Expr, 1) }), curr.Next);

                    for (curr = newCallCmd.Attributes; curr.Next != null; curr = curr.Next)
                    {
                        if (curr.Next.Key.StartsWith("arg"))
                        {
                            curr.Next = new QKeyValue(Token.NoToken, curr.Next.Key, new List<object>(new object[] { Dualise(curr.Next.Params[0] as Expr, 1) }), curr.Next.Next);
                        }
                    }
                }
                else if (newCallCmd.callee.StartsWith("_CHECK_ATOMIC"))
                {
                    QKeyValue curr = newCallCmd.Attributes;
                    if (curr.Key.StartsWith("arg"))
                        newCallCmd.Attributes = new QKeyValue(Token.NoToken, curr.Key, new List<object>(new object[] { Dualise(curr.Params[0] as Expr, 2) }), curr.Next);

                    for (curr = newCallCmd.Attributes; curr.Next != null; curr = curr.Next)
                    {
                        if (curr.Next.Key.StartsWith("arg"))
                        {
                            curr.Next = new QKeyValue(Token.NoToken, curr.Next.Key, new List<object>(new object[] { Dualise(curr.Next.Params[0] as Expr, 2) }), curr.Next.Next);
                        }
                    }
                }

                cs.Add(newCallCmd);

                if (GPUVerifier.IsBarrier(call.Proc))
                {
                    foreach (var biDescriptor in barrierInvariantDescriptors)
                    {
                        foreach (var instantiation in biDescriptor.GetInstantiationCmds())
                            cs.Add(instantiation);
                    }

                    barrierInvariantDescriptors.Clear();
                }
            }
            else if (c is AssignCmd)
            {
                AssignCmd assign = c as AssignCmd;

                var vd1 = new VariableDualiser(1, Verifier, procName);
                var vd2 = new VariableDualiser(2, Verifier, procName);

                List<AssignLhs> lhss1 = new List<AssignLhs>();
                List<AssignLhs> lhss2 = new List<AssignLhs>();

                List<Expr> rhss1 = new List<Expr>();
                List<Expr> rhss2 = new List<Expr>();

                foreach (var pair in assign.Lhss.Zip(assign.Rhss))
                {
                    if (pair.Item1 is SimpleAssignLhs
                        && Verifier.UniformityAnalyser.IsUniform(
                            procName, (pair.Item1 as SimpleAssignLhs).AssignedVariable.Name))
                    {
                        lhss1.Add(pair.Item1);
                        rhss1.Add(pair.Item2);
                    }
                    else
                    {
                        lhss1.Add(vd1.Visit(pair.Item1.Clone() as AssignLhs) as AssignLhs);
                        lhss2.Add(vd2.Visit(pair.Item1.Clone() as AssignLhs) as AssignLhs);
                        rhss1.Add(vd1.VisitExpr(pair.Item2.Clone() as Expr));
                        rhss2.Add(vd2.VisitExpr(pair.Item2.Clone() as Expr));
                    }
                }

                Debug.Assert(lhss1.Count > 0);
                cs.Add(new AssignCmd(Token.NoToken, lhss1, rhss1));

                if (lhss2.Count > 0)
                {
                    cs.Add(new AssignCmd(Token.NoToken, lhss2, rhss2));
                }
            }
            else if (c is HavocCmd)
            {
                HavocCmd havoc = c as HavocCmd;
                Debug.Assert(havoc.Vars.Count() == 1);

                HavocCmd newHavoc;

                var idents = new List<IdentifierExpr>
                {
                    (IdentifierExpr)new VariableDualiser(1, Verifier, procName).VisitIdentifierExpr(havoc.Vars[0].Clone() as IdentifierExpr),
                    (IdentifierExpr)new VariableDualiser(2, Verifier, procName).VisitIdentifierExpr(havoc.Vars[0].Clone() as IdentifierExpr)
                };
                newHavoc = new HavocCmd(havoc.tok, idents);

                cs.Add(newHavoc);
            }
            else if (c is AssertCmd)
            {
                AssertCmd a = c as AssertCmd;

                if (QKeyValue.FindBoolAttribute(a.Attributes, "sourceloc")
                  || QKeyValue.FindBoolAttribute(a.Attributes, "block_sourceloc")
                  || QKeyValue.FindBoolAttribute(a.Attributes, "array_bounds"))
                {
                    // This is just a location marker, so we do not dualise it
                    cs.Add(new AssertCmd(
                        Token.NoToken,
                        new VariableDualiser(1, Verifier, procName).VisitExpr(a.Expr.Clone() as Expr),
                        (QKeyValue)a.Attributes.Clone()));
                }
                else
                {
                    var isUniform = Verifier.UniformityAnalyser.IsUniform(procName, a.Expr);
                    cs.Add(MakeThreadSpecificAssert(a, 1));
                    if (!GPUVerifyVCGenCommandLineOptions.AsymmetricAsserts && !ContainsAsymmetricExpression(a.Expr) && !isUniform)
                    {
                        cs.Add(MakeThreadSpecificAssert(a, 2));
                    }
                }
            }
            else if (c is AssumeCmd)
            {
                AssumeCmd ass = c as AssumeCmd;

                if (QKeyValue.FindStringAttribute(ass.Attributes, "captureState") != null)
                {
                    cs.Add(c);
                }
                else if (QKeyValue.FindBoolAttribute(ass.Attributes, "backedge"))
                {
                    AssumeCmd newAss = new AssumeCmd(
                        c.tok,
                        Expr.Or(
                            new VariableDualiser(1, Verifier, procName).VisitExpr(ass.Expr.Clone() as Expr),
                            new VariableDualiser(2, Verifier, procName).VisitExpr(ass.Expr.Clone() as Expr)));
                    newAss.Attributes = ass.Attributes;
                    cs.Add(newAss);
                }
                else if (QKeyValue.FindBoolAttribute(ass.Attributes, "atomic_refinement"))
                {
                    // Generate the following:
                    // havoc v$1, v$2;
                    // assume !_USED[offset$1][v$1];
                    // _USED[offset$1][v$1] := true;
                    // assume !_USED[offset$2][v$2];
                    // _USED[offset$2][v$2] := true;
                    Expr variable = QKeyValue.FindExprAttribute(ass.Attributes, "variable");
                    Expr offset = QKeyValue.FindExprAttribute(ass.Attributes, "offset");

                    List<Expr> offsets = Enumerable.Range(1, 2).Select(x => new VariableDualiser(x, Verifier, procName).VisitExpr(offset.Clone() as Expr)).ToList();
                    List<Expr> vars = Enumerable.Range(1, 2).Select(x => new VariableDualiser(x, Verifier, procName).VisitExpr(variable.Clone() as Expr)).ToList();
                    IdentifierExpr arrayref = new IdentifierExpr(Token.NoToken, Verifier.FindOrCreateUsedMap(QKeyValue.FindStringAttribute(ass.Attributes, "arrayref"), vars[0].Type));

                    foreach (int i in Enumerable.Range(0, 2))
                    {
                        var select = new NAryExpr(
                            Token.NoToken, new MapSelect(Token.NoToken, 1), new List<Expr> { arrayref, offsets[i] });
                        select = new NAryExpr(
                            Token.NoToken, new MapSelect(Token.NoToken, 1), new List<Expr> { select, vars[i] });
                        AssumeCmd newAss = new AssumeCmd(c.tok, Expr.Not(select));
                        cs.Add(newAss);

                        var lhs = new MapAssignLhs(
                            Token.NoToken, new SimpleAssignLhs(Token.NoToken, arrayref), new List<Expr> { offsets[i] });
                        lhs = new MapAssignLhs(
                            Token.NoToken, lhs, new List<Expr> { vars[i] });
                        AssignCmd assign = new AssignCmd(
                            c.tok, new List<AssignLhs> { lhs }, new List<Expr> { Expr.True });
                        cs.Add(assign);
                    }
                }
                else
                {
                    var isUniform = Verifier.UniformityAnalyser.IsUniform(procName, ass.Expr);
                    AssumeCmd newAss = new AssumeCmd(c.tok, new VariableDualiser(1, Verifier, procName).VisitExpr(ass.Expr.Clone() as Expr));
                    if (!ContainsAsymmetricExpression(ass.Expr) && !isUniform)
                    {
                        newAss.Expr = Expr.And(newAss.Expr, new VariableDualiser(2, Verifier, procName).VisitExpr(ass.Expr.Clone() as Expr));
                    }

                    newAss.Attributes = ass.Attributes;
                    cs.Add(newAss);
                }
            }
            else
            {
                Debug.Assert(false);
            }
        }

        private Block MakeDual(Block b)
        {
            var newCmds = new List<Cmd>();
            foreach (Cmd c in b.Cmds)
            {
                MakeDual(newCmds, c);
            }

            b.Cmds = newCmds;
            return b;
        }

        private List<PredicateCmd> MakeDualInvariants(List<PredicateCmd> originalInvariants)
        {
            List<PredicateCmd> result = new List<PredicateCmd>();
            foreach (PredicateCmd p in originalInvariants)
            {
                {
                    PredicateCmd newP = new AssertCmd(p.tok, Dualise(p.Expr, 1));
                    newP.Attributes = p.Attributes;
                    result.Add(newP);
                }

                if (!ContainsAsymmetricExpression(p.Expr)
                    && !Verifier.UniformityAnalyser.IsUniform(procName, p.Expr))
                {
                    PredicateCmd newP = new AssertCmd(p.tok, Dualise(p.Expr, 2));
                    newP.Attributes = p.Attributes;
                    result.Add(newP);
                }
            }

            return result;
        }

        private void MakeDualLocalVariables(Implementation impl)
        {
            List<Variable> newLocalVars = new List<Variable>();

            foreach (LocalVariable v in impl.LocVars)
            {
                if (Verifier.UniformityAnalyser.IsUniform(procName, v.Name))
                {
                    newLocalVars.Add(v);
                }
                else
                {
                    newLocalVars.Add(
                        new VariableDualiser(1, Verifier, procName).VisitVariable(v.Clone() as Variable));
                    newLocalVars.Add(
                        new VariableDualiser(2, Verifier, procName).VisitVariable(v.Clone() as Variable));
                }
            }

            impl.LocVars = newLocalVars;
        }

        private static bool ContainsAsymmetricExpression(Expr expr)
        {
            AsymmetricExpressionFinder finder = new AsymmetricExpressionFinder();
            finder.VisitExpr(expr);
            return finder.FoundAsymmetricExpr();
        }

        private List<Variable> DualiseVariableSequence(List<Variable> seq)
        {
            List<Variable> uniform = new List<Variable>();
            List<Variable> nonuniform = new List<Variable>();

            foreach (Variable v in seq)
            {
                if (Verifier.UniformityAnalyser.IsUniform(procName, v.Name))
                {
                    uniform.Add(v);
                }
                else
                {
                    nonuniform.Add(new VariableDualiser(1, Verifier, procName).VisitVariable((Variable)v.Clone()));
                }
            }

            foreach (Variable v in seq)
            {
                if (!Verifier.UniformityAnalyser.IsUniform(procName, v.Name))
                {
                    nonuniform.Add(new VariableDualiser(2, Verifier, procName).VisitVariable((Variable)v.Clone()));
                }
            }

            List<Variable> result = uniform;
            result.AddRange(nonuniform);
            return result;
        }

        private void DualiseImplementation(Implementation impl)
        {
            procName = impl.Name;
            impl.InParams = DualiseVariableSequence(impl.InParams);
            impl.OutParams = DualiseVariableSequence(impl.OutParams);
            MakeDualLocalVariables(impl);
            impl.Blocks = new List<Block>(impl.Blocks.Select(MakeDual));
            procName = null;
        }

        private Expr Dualise(Expr expr, int thread)
        {
            return new VariableDualiser(thread, Verifier, procName).VisitExpr(expr.Clone() as Expr);
        }

        private int UpdateDeclarationsAndCountTotal(List<Declaration> decls)
        {
            var newDecls = Verifier.Program.TopLevelDeclarations.Where(d => !decls.Contains(d));
            decls.AddRange(newDecls.ToList());
            return decls.Count();
        }

        public void DualiseKernel()
        {
            List<Declaration> newTopLevelDeclarations = new List<Declaration>();

            // This loop really does have to be a "for(i ...)" loop.  The reason is
            // that dualisation may add additional functions to the program, which
            // get put into the program's top level declarations and also need to
            // be dualised.
            var decls = Verifier.Program.TopLevelDeclarations.ToList();
            for (int i = 0; i < UpdateDeclarationsAndCountTotal(decls); i++)
            {
                Declaration d = decls[i];

                if (d is Axiom)
                {
                    VariableDualiser vd1 = new VariableDualiser(1, Verifier, null);
                    VariableDualiser vd2 = new VariableDualiser(2, Verifier, null);
                    Axiom newAxiom1 = vd1.VisitAxiom(d.Clone() as Axiom);
                    Axiom newAxiom2 = vd2.VisitAxiom(d.Clone() as Axiom);
                    newTopLevelDeclarations.Add(newAxiom1);

                    // Test whether dualisation had any effect by seeing whether the new
                    // axioms are syntactically indistinguishable.  If they are, then there
                    // is no point adding the second axiom.
                    if (!newAxiom1.ToString().Equals(newAxiom2.ToString()))
                        newTopLevelDeclarations.Add(newAxiom2);

                    continue;
                }

                if (d is Procedure)
                {
                    DualiseProcedure(d as Procedure);
                    newTopLevelDeclarations.Add(d);
                    continue;
                }

                if (d is Implementation)
                {
                    DualiseImplementation(d as Implementation);
                    newTopLevelDeclarations.Add(d);
                    continue;
                }

                if (d is Variable
                    && ((d as Variable).IsMutable
                        || Verifier.IsThreadLocalIdConstant(d as Variable)
                        || (Verifier.IsGroupIdConstant(d as Variable) && !GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)))
                {
                    var v = d as Variable;

                    if (v.Name.Contains("_NOT_ACCESSED_") || v.Name.Contains("_ARRAY_OFFSET"))
                    {
                        newTopLevelDeclarations.Add(v);
                        continue;
                    }

                    if (QKeyValue.FindBoolAttribute(v.Attributes, "atomic_usedmap"))
                    {
                        newTopLevelDeclarations.Add(v);
                        continue;
                    }

                    if (Verifier.KernelArrayInfo.GetGlobalArrays(true).Contains(v))
                    {
                        newTopLevelDeclarations.Add(v);
                        continue;
                    }

                    if (Verifier.KernelArrayInfo.GetGroupSharedArrays(true).Contains(v))
                    {
                        if (!GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
                        {
                            var type = new MapType(
                                Token.NoToken,
                                new List<TypeVariable>(),
                                new List<Microsoft.Boogie.Type> { Microsoft.Boogie.Type.GetBvType(1) },
                                v.TypedIdent.Type);
                            Variable newV = new GlobalVariable(
                                Token.NoToken, new TypedIdent(Token.NoToken, v.Name, type));
                            newV.Attributes = v.Attributes;
                            newTopLevelDeclarations.Add(newV);
                        }
                        else
                        {
                            newTopLevelDeclarations.Add(v);
                        }

                        continue;
                    }

                    newTopLevelDeclarations.Add(new VariableDualiser(1, Verifier, null).VisitVariable((Variable)v.Clone()));
                    if (!QKeyValue.FindBoolAttribute(v.Attributes, "race_checking"))
                    {
                        newTopLevelDeclarations.Add(new VariableDualiser(2, Verifier, null).VisitVariable((Variable)v.Clone()));
                    }

                    continue;
                }

                newTopLevelDeclarations.Add(d);
            }

            Verifier.Program.TopLevelDeclarations = newTopLevelDeclarations;
        }
    }
}
