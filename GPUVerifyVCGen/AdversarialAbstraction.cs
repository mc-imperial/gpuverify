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
    using Microsoft.Boogie;

    public class AdversarialAbstraction
    {
        private GPUVerifier verifier;

        private List<Variable> newLocalVars = null;
        private int abstractedCallArgCounter = 0;

        public AdversarialAbstraction(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public void Abstract()
        {
            List<Declaration> newTopLevelDeclarations = new List<Declaration>();
            foreach (Declaration d in verifier.Program.TopLevelDeclarations)
            {
                // Note that we do still need to abstract arrays, even if we have disabled race checking for them
                if (d is Variable
                    && verifier.KernelArrayInfo.ContainsGlobalOrGroupSharedArray((Variable)d, true)
                    && verifier.ArrayModelledAdversarially((Variable)d))
                {
                    continue;
                }

                if (d is Implementation)
                    Abstract((Implementation)d);

                if (d is Procedure)
                    Abstract((Procedure)d);

                newTopLevelDeclarations.Add(d);
            }

            verifier.Program.TopLevelDeclarations = newTopLevelDeclarations;
        }

        private void AbstractRequiresClauses(Procedure proc)
        {
            List<Requires> newRequires = new List<Requires>();
            foreach (Requires r in proc.Requires)
            {
                var visitor = new AccessesAdversarialArrayVisitor(verifier);
                visitor.VisitRequires(r);
                if (!visitor.Found)
                    newRequires.Add(r);
            }

            proc.Requires = newRequires;
        }

        private void Abstract(Procedure proc)
        {
            AbstractModifiesSet(proc);
            AbstractRequiresClauses(proc);
        }

        private void AbstractModifiesSet(Procedure proc)
        {
            List<IdentifierExpr> newModifies = new List<IdentifierExpr>();
            foreach (IdentifierExpr e in proc.Modifies)
            {
                var visitor = new AccessesAdversarialArrayVisitor(verifier);
                visitor.VisitIdentifierExpr(e);
                if (!visitor.Found)
                    newModifies.Add(e);
            }

            proc.Modifies = newModifies;
        }

        private void Abstract(Implementation impl)
        {
            newLocalVars = new List<Variable>();
            abstractedCallArgCounter = 0;
            foreach (Variable v in impl.LocVars)
            {
                Debug.Assert(!verifier.KernelArrayInfo.GetGroupSharedArrays(true).Contains(v));
                newLocalVars.Add(v);
            }

            impl.LocVars = newLocalVars;
            impl.Blocks = impl.Blocks.Select(Abstract).ToList();
            newLocalVars = null;
        }

        private Block Abstract(Block b)
        {
            var newCmds = new List<Cmd>();

            foreach (Cmd c in b.Cmds)
            {
                if (c is CallCmd)
                {
                    var call = c as CallCmd;

                    if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic"))
                    {
                        // Discard the call
                        Debug.Assert(call.Ins.Count >= 1);
                        var ie = call.Ins[0] as IdentifierExpr;
                        Debug.Assert(ie != null);
                        Debug.Assert(verifier.ArrayModelledAdversarially(ie.Decl));
                        continue;
                    }

                    for (int i = 0; i < call.Ins.Count; i++)
                    {
                        ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
                        rc.Visit(call.Ins[i]);
                        bool foundAdversarial = false;
                        foreach (AccessRecord ar in rc.NonPrivateAccesses)
                        {
                            if (verifier.ArrayModelledAdversarially(ar.V))
                            {
                                foundAdversarial = true;
                                break;
                            }
                        }

                        if (foundAdversarial)
                        {
                            LocalVariable lv = new LocalVariable(
                                Token.NoToken,
                                new TypedIdent(Token.NoToken, "_abstracted_call_arg_" + abstractedCallArgCounter, call.Ins[i].Type));
                            abstractedCallArgCounter++;
                            newLocalVars.Add(lv);
                            newCmds.Add(new HavocCmd(
                                Token.NoToken,
                                new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(Token.NoToken, lv) })));
                            call.Ins[i] = new IdentifierExpr(Token.NoToken, lv);
                        }
                    }
                }

                if (c is AssignCmd)
                {
                    AssignCmd assign = c as AssignCmd;

                    var lhss = new List<AssignLhs>();
                    var rhss = new List<Expr>();

                    foreach (var lhsRhs in assign.Lhss.Zip(assign.Rhss))
                    {
                        AssignLhs lhs = lhsRhs.Item1;
                        Expr rhs = lhsRhs.Item2;
                        ReadCollector rc = new ReadCollector(verifier.KernelArrayInfo);
                        rc.Visit(rhs);

                        bool foundAdversarial = false;
                        foreach (AccessRecord ar in rc.NonPrivateAccesses)
                        {
                            if (verifier.ArrayModelledAdversarially(ar.V))
                            {
                                foundAdversarial = true;
                                break;
                            }
                        }

                        if (foundAdversarial)
                        {
                            Debug.Assert(lhs is SimpleAssignLhs);
                            newCmds.Add(new HavocCmd(c.tok, new List<IdentifierExpr>(new IdentifierExpr[] { (lhs as SimpleAssignLhs).AssignedVariable })));
                            continue;
                        }

                        WriteCollector wc = new WriteCollector(verifier.KernelArrayInfo);
                        wc.Visit(lhs);
                        if (wc.FoundNonPrivateWrite())
                        {
                            if (verifier.ArrayModelledAdversarially(wc.GetAccess().V))
                            {
                                continue; // Just remove the write
                            }
                        }

                        lhss.Add(lhs);
                        rhss.Add(rhs);
                    }

                    if (lhss.Count != 0)
                    {
                        newCmds.Add(new AssignCmd(assign.tok, lhss, rhss));
                    }

                    continue;
                }

                newCmds.Add(c);
            }

            b.Cmds = newCmds;
            return b;
        }

        private class AccessesAdversarialArrayVisitor : StandardVisitor
        {
            private GPUVerifier verifier;

            public AccessesAdversarialArrayVisitor(GPUVerifier verifier)
            {
                this.verifier = verifier;
                this.Found = false;
            }

            public bool Found { get; private set; }

            public override Variable VisitVariable(Variable v)
            {
                if (verifier.KernelArrayInfo.ContainsGlobalOrGroupSharedArray(v, true))
                {
                    if (verifier.ArrayModelledAdversarially(v))
                    {
                        Found = true;
                    }
                }

                return base.VisitVariable(v);
            }
        }
    }
}
