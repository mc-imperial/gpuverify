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

    public class UnaryBarrierInvariantDescriptor : BarrierInvariantDescriptor
    {
        private List<Expr> instantiationExprs;

        public UnaryBarrierInvariantDescriptor(
            Expr predicate, Expr barrierInvariant, QKeyValue sourceLocationInfo, KernelDualiser dualiser, string procName, GPUVerifier verifier)
            : base(predicate, barrierInvariant, sourceLocationInfo, dualiser, procName, verifier)
        {
            instantiationExprs = new List<Expr>();
        }

        public void AddInstantiationExpr(Expr instantiationExpr)
        {
            instantiationExprs.Add(instantiationExpr);
        }

        public override List<AssumeCmd> GetInstantiationCmds()
        {
            var result = new List<AssumeCmd>();
            foreach (var instantiation in instantiationExprs)
            {
                foreach (var thread in Enumerable.Range(1, 2))
                {
                    var vd = new VariableDualiser(thread, Dualiser.Verifier, ProcName);
                    var ti = new ThreadInstantiator(instantiation, thread, Dualiser.Verifier, ProcName);

                    var assume = new AssumeCmd(
                        Token.NoToken,
                        Expr.Imp(
                            vd.VisitExpr(Predicate),
                            Expr.Imp(
                                Expr.And(
                                    NonNegative(instantiation),
                                    NotTooLarge(instantiation)),
                                ti.VisitExpr(BarrierInvariant))));
                    result.Add(vd.VisitAssumeCmd(assume) as AssumeCmd);
                }
            }

            return result;
        }

        private class ThreadInstantiator : Duplicator
        {
            private Expr instantiationExpr;
            private int thread;
            private GPUVerifier verifier;
            private UniformityAnalyser uni;
            private string procName;

            public ThreadInstantiator(
                Expr instantiationExpr, int thread, GPUVerifier verifier, string procName)
            {
                this.instantiationExpr = instantiationExpr;
                this.thread = thread;
                this.verifier = verifier;
                this.uni = verifier.UniformityAnalyser;
                this.procName = procName;
            }

            public override Expr VisitIdentifierExpr(IdentifierExpr node)
            {
                Debug.Assert(!(node.Decl is Formal));

                if (verifier.IsThreadLocalIdConstant(node.Decl))
                {
                    Debug.Assert(node.Decl.Name.Equals(verifier.IdX.Name));
                    return instantiationExpr.Clone() as Expr;
                }

                if (node.Decl is Constant
                    || QKeyValue.FindBoolAttribute(node.Decl.Attributes, "global")
                    || QKeyValue.FindBoolAttribute(node.Decl.Attributes, "group_shared")
                    || (uni != null && uni.IsUniform(procName, node.Decl.Name)))
                {
                    return base.VisitIdentifierExpr(node);
                }

                Console.WriteLine("Expression " + node + " is not valid as part of a barrier invariant: it cannot be instantiated by arbitrary threads.");
                Console.WriteLine("Check that it is not a thread local variable, or a thread local (rather than __local or __global) array.");
                Console.WriteLine("In particular, if you have a local variable called tid, which you initialise to e.g. get_local_id(0), this will not work:");
                Console.WriteLine("  you need to use get_local_id(0) directly.");
                Environment.Exit(1);
                return null;
            }

            private bool InstantiationExprIsThreadId()
            {
                return (instantiationExpr is IdentifierExpr)
                    && ((IdentifierExpr)instantiationExpr).Decl.Name.Equals(verifier.MakeThreadId("X", thread).Name);
            }
        }
    }
}
