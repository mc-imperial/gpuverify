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
    using Microsoft.Boogie;

    internal class BinaryBarrierInvariantDescriptor : BarrierInvariantDescriptor
    {
        private List<Tuple<Expr, Expr>> instantiationExprPairs;

        public BinaryBarrierInvariantDescriptor(Expr predicate, Expr barrierInvariant,
                QKeyValue sourceLocationInfo,
                KernelDualiser dualiser, string procName, GPUVerifier verifier)
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
            result.Expr = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), result.Expr);
            return result;
        }

        public override List<AssumeCmd> GetInstantiationCmds()
        {
            var result = new List<AssumeCmd>();
            foreach (var instantiation in instantiationExprPairs)
            {
                foreach (var thread in new int[] { 1, 2 })
                {
                    var vd = new VariableDualiser(thread, Dualiser.verifier.uniformityAnalyser, ProcName);
                    var ti = new ThreadPairInstantiator(Dualiser.verifier, instantiation.Item1, instantiation.Item2, thread);

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
    }
}
