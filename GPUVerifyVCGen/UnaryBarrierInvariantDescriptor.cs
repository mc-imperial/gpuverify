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
  class UnaryBarrierInvariantDescriptor : BarrierInvariantDescriptor {
    private List<Expr> InstantiationExprs;

    public UnaryBarrierInvariantDescriptor(Expr Predicate, Expr BarrierInvariant,
        QKeyValue SourceLocationInfo, KernelDualiser Dualiser, string ProcName,
        GPUVerifier Verifier) :
      base(Predicate, BarrierInvariant, SourceLocationInfo, Dualiser, ProcName, Verifier) {
Console.WriteLine(BarrierInvariant);
      InstantiationExprs = new List<Expr>();
    }

    public void AddInstantiationExpr(Expr InstantiationExpr) {
      InstantiationExprs.Add(InstantiationExpr);
    }

    internal override List<AssumeCmd> GetInstantiationCmds() {
      var result = new List<AssumeCmd>();
      foreach (var Instantiation in InstantiationExprs) {
        foreach (var Thread in new int[] { 1, 2 }) {
          var vd = new VariableDualiser(Thread, Dualiser.verifier.uniformityAnalyser, ProcName);
          var ti = new ThreadInstantiator(Instantiation, Thread,
            Dualiser.verifier.uniformityAnalyser, ProcName);

          var Assume = new AssumeCmd(Token.NoToken,
            Expr.Imp(vd.VisitExpr(Predicate),
              Expr.Imp(Expr.And(
                NonNegative(Instantiation),
                NotTooLarge(Instantiation)),
              ti.VisitExpr(BarrierInvariant))));
          result.Add(vd.VisitAssumeCmd(Assume) as AssumeCmd);
        }
      }
      return result;
    }

  }
}
