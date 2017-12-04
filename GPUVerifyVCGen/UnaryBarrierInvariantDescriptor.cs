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
    using Microsoft.Boogie;

    internal class UnaryBarrierInvariantDescriptor : BarrierInvariantDescriptor
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
                    var vd = new VariableDualiser(thread, Dualiser.verifier.uniformityAnalyser, ProcName);
                    var ti = new ThreadInstantiator(
                        instantiation, thread, Dualiser.verifier.uniformityAnalyser, ProcName);

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
    }
}
