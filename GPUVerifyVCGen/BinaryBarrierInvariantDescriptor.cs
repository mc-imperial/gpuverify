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

namespace GPUVerify {
  class BinaryBarrierInvariantDescriptor : BarrierInvariantDescriptor {

    private List<Tuple<Expr, Expr>> InstantiationExprPairs;

    public BinaryBarrierInvariantDescriptor(Expr Predicate, Expr BarrierInvariant,
        QKeyValue SourceLocationInfo,
        KernelDualiser Dualiser, string ProcName, GPUVerifier Verifier) :
      base(Predicate, BarrierInvariant, SourceLocationInfo, Dualiser, ProcName, Verifier) {
      InstantiationExprPairs = new List<Tuple<Expr, Expr>>();
    }

    public void AddInstantiationExprPair(Expr first, Expr second) {
      InstantiationExprPairs.Add(new Tuple<Expr, Expr>(first, second));
    }

    internal override AssertCmd GetAssertCmd() {
      AssertCmd result = base.GetAssertCmd();
      result.Expr = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), result.Expr);
      return result;
    }

    internal override List<AssumeCmd> GetInstantiationCmds() {
      var result = new List<AssumeCmd>();
      foreach (var Instantiation in InstantiationExprPairs) {
        foreach (var Thread in new int[] { 1, 2 }) {

          var vd = new VariableDualiser(Thread, Dualiser.verifier.uniformityAnalyser, ProcName);

          var ThreadInstantiationExpr = vd.VisitExpr(Instantiation.Item1);
          var OtherThreadInstantiationExpr = vd.VisitExpr(Instantiation.Item2);

          var ti = new ThreadPairInstantiator(Dualiser.verifier, ThreadInstantiationExpr, OtherThreadInstantiationExpr, Thread);

          result.Add(new AssumeCmd(
            Token.NoToken,
            Expr.Imp(vd.VisitExpr(Predicate),
              Expr.Imp(
                Expr.And(
                  Expr.And(
                    Expr.And(NonNegative(ThreadInstantiationExpr),
                             NotTooLarge(ThreadInstantiationExpr)),
                    Expr.And(NonNegative(OtherThreadInstantiationExpr),
                             NotTooLarge(OtherThreadInstantiationExpr))
                  ),
                  Expr.Neq(ThreadInstantiationExpr, OtherThreadInstantiationExpr)),
              ti.VisitExpr(BarrierInvariant)))));
        }
      }
      return result;
    }


  }
}
