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
using Microsoft.Basetypes;
using Microsoft.Boogie;

namespace GPUVerify {

class StrideConstraint {

  public static StrideConstraint Bottom(GPUVerifier verifier, Expr e) {
    return new ModStrideConstraint(verifier.IntRep.GetLiteral(1, e.Type is BvType ? e.Type.BvBits : verifier.size_t_bits),
                                   verifier.IntRep.GetLiteral(0, e.Type is BvType ? e.Type.BvBits : verifier.size_t_bits));
  }

  public bool IsBottom() {
    var msc = this as ModStrideConstraint;
    if (msc == null)
        return false;

    var le = msc.mod as LiteralExpr;
    if (le == null)
        return false;

    var bvc = le.Val as BvConst;
    if (bvc == null)
        return false;

    return bvc.Value.InInt32 && bvc.Value.ToInt == 1;
  }

  public Expr MaybeBuildPredicate(GPUVerifier verifier, Expr e) {
    var msc = this as ModStrideConstraint;
    if (msc != null && !msc.IsBottom()) {
      Expr modEqExpr = Expr.Eq(verifier.IntRep.MakeModPow2(e, msc.mod), verifier.IntRep.MakeModPow2(msc.modEq, msc.mod));
      return modEqExpr;
    }

    return null;
  }

  private static StrideConstraint BuildAddStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc) {
    if (lhsc is EqStrideConstraint && rhsc is EqStrideConstraint) {
      return new EqStrideConstraint(e);
    }

    if (lhsc is EqStrideConstraint && rhsc is ModStrideConstraint)
      return BuildAddStrideConstraint(verifier, e, rhsc, lhsc);

    if (lhsc is ModStrideConstraint && rhsc is EqStrideConstraint) {
      var lhsmc = (ModStrideConstraint)lhsc;
      var rhsec = (EqStrideConstraint)rhsc;

      return new ModStrideConstraint(lhsmc.mod, verifier.IntRep.MakeAdd(lhsmc.modEq, rhsec.eq));
    }

    if (lhsc is ModStrideConstraint && rhsc is ModStrideConstraint) {
      var lhsmc = (ModStrideConstraint)lhsc;
      var rhsmc = (ModStrideConstraint)rhsc;

      if (lhsmc.mod == rhsmc.mod)
        return new ModStrideConstraint(lhsmc.mod, verifier.IntRep.MakeAdd(lhsmc.modEq, rhsmc.modEq));
    }

    return Bottom(verifier, e);
  }

  private static StrideConstraint BuildMulStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc) {
    if (lhsc is EqStrideConstraint && rhsc is EqStrideConstraint) {
      return new EqStrideConstraint(e);
    }

    if (lhsc is EqStrideConstraint && rhsc is ModStrideConstraint)
      return BuildMulStrideConstraint(verifier, e, rhsc, lhsc);

    if (lhsc is ModStrideConstraint && rhsc is EqStrideConstraint) {
      var lhsmc = (ModStrideConstraint)lhsc;
      var rhsec = (EqStrideConstraint)rhsc;

      return new ModStrideConstraint(verifier.IntRep.MakeMul(lhsmc.mod, rhsec.eq),
                                     verifier.IntRep.MakeMul(lhsmc.modEq, rhsec.eq));
    }

    return Bottom(verifier, e);
  }

  public static StrideConstraint FromExpr(GPUVerifier verifier, Implementation impl, Expr e) {
    if (e is LiteralExpr)
      return new EqStrideConstraint(e);

    var ie = e as IdentifierExpr;
    if (ie != null) {
      if(GPUVerifier.IsConstantInCurrentRegion(ie))
        return new EqStrideConstraint(e);

      var rsa = verifier.reducedStrengthAnalyses[impl];
      var sc = rsa.GetStrideConstraint(ie.Decl.Name);
      if (sc == null)
        return Bottom(verifier, e);
      return sc;
    }

    Expr lhs, rhs;

    if (verifier.IntRep.IsAdd(e, out lhs, out rhs)) {
      var lhsc = FromExpr(verifier, impl, lhs);
      var rhsc = FromExpr(verifier, impl, rhs);
      return BuildAddStrideConstraint(verifier, e, lhsc, rhsc);
    }

    if (verifier.IntRep.IsMul(e, out lhs, out rhs)) {
      var lhsc = FromExpr(verifier, impl, lhs);
      var rhsc = FromExpr(verifier, impl, rhs);
      return BuildMulStrideConstraint(verifier, e, lhsc, rhsc);
    }

    return Bottom(verifier, e);
  }

}

class EqStrideConstraint : StrideConstraint {
  public EqStrideConstraint(Expr eq) { this.eq = eq; }
  public Expr eq;
}

class ModStrideConstraint : StrideConstraint {
  public ModStrideConstraint(Expr mod, Expr modEq) { this.mod = mod; this.modEq = modEq; }
  public Expr mod, modEq;
}

}
