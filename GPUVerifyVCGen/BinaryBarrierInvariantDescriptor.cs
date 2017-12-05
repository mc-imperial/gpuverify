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

    public class BinaryBarrierInvariantDescriptor : BarrierInvariantDescriptor
    {
        private List<Tuple<Expr, Expr>> instantiationExprPairs;

        public BinaryBarrierInvariantDescriptor(
            Expr predicate, Expr barrierInvariant, QKeyValue sourceLocationInfo, KernelDualiser dualiser, string procName, GPUVerifier verifier)
            : base(predicate, barrierInvariant, sourceLocationInfo, dualiser, procName, verifier)
        {
            instantiationExprPairs = new List<Tuple<Expr, Expr>>();
        }

        public void AddInstantiationExprPair(Expr first, Expr second)
        {
            instantiationExprPairs.Add(new Tuple<Expr, Expr>(first, second));
        }

        public override AssertCmd GetAssertCmd()
        {
            AssertCmd result = base.GetAssertCmd();
            result.Expr = Expr.Imp(Verifier.ThreadsInSameGroup(), result.Expr);
            return result;
        }

        public override List<AssumeCmd> GetInstantiationCmds()
        {
            var result = new List<AssumeCmd>();
            foreach (var instantiation in instantiationExprPairs)
            {
                foreach (var thread in new int[] { 1, 2 })
                {
                    var vd = new VariableDualiser(thread, Dualiser.Verifier, ProcName);
                    var ti = new ThreadPairInstantiator(Dualiser.Verifier, instantiation.Item1, instantiation.Item2, thread);

                    var assume = new AssumeCmd(
                        Token.NoToken,
                        Expr.Imp(
                            vd.VisitExpr(Predicate),
                            Expr.Imp(
                                Expr.And(
                                    Expr.And(
                                        Expr.And(
                                            NonNegative(instantiation.Item1),
                                            NotTooLarge(instantiation.Item1)),
                                        Expr.And(
                                            NonNegative(instantiation.Item2),
                                            NotTooLarge(instantiation.Item2))),
                                    Expr.Neq(instantiation.Item1, instantiation.Item2)),
                                ti.VisitExpr(BarrierInvariant))));
                    result.Add(vd.VisitAssumeCmd(assume) as AssumeCmd);
                }
            }

            return result;
        }

        private class ThreadPairInstantiator : Duplicator
        {
            private GPUVerifier verifier;
            private Tuple<Expr, Expr> instantiationExprs;
            private int thread;

            public ThreadPairInstantiator(GPUVerifier verifier, Expr instantiationExpr1, Expr instantiationExpr2, int thread)
            {
                this.verifier = verifier;
                this.instantiationExprs = new Tuple<Expr, Expr>(instantiationExpr1, instantiationExpr2);
                this.thread = thread;
            }

            public override Expr VisitIdentifierExpr(IdentifierExpr node)
            {
                Debug.Assert(!(node.Decl is Formal));

                if (verifier.IsThreadLocalIdConstant(node.Decl))
                {
                    Debug.Assert(node.Decl.Name.Equals(verifier.IdX.Name));
                    return instantiationExprs.Item1.Clone() as Expr;
                }

                if (node.Decl is Constant
                    || verifier.KernelArrayInfo.GetGroupSharedArrays(true).Contains(node.Decl)
                    || verifier.KernelArrayInfo.GetGlobalArrays(true).Contains(node.Decl))
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

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                if (node.Fun is FunctionCall)
                {
                    FunctionCall call = node.Fun as FunctionCall;

                    // Alternate instantiation order for "other thread" functions.
                    // Note that we do not alternate the "Thread" field, as we are not switching the
                    // thread for which instantiation is being performed
                    if (VariableDualiser.OtherFunctionNames.Contains(call.Func.Name))
                    {
                        return new ThreadPairInstantiator(verifier, instantiationExprs.Item2, instantiationExprs.Item1, thread)
                          .VisitExpr(node.Args[0]);
                    }
                }

                return base.VisitNAryExpr(node);
            }
        }
    }
}
