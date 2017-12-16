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
    using Microsoft.Boogie;

    public class StrideConstraint
    {
        public static StrideConstraint Bottom(GPUVerifier verifier, Expr e)
        {
            return new ModStrideConstraint(
                verifier.IntRep.GetLiteral(1, e.Type),
                verifier.IntRep.GetZero(e.Type));
        }

        public bool IsBottom()
        {
            var msc = this as ModStrideConstraint;
            if (msc == null)
                return false;

            var le = msc.Mod as LiteralExpr;
            if (le == null)
                return false;

            var bvc = le.Val as BvConst;
            if (bvc == null)
                return false;

            return bvc.Value.InInt32 && bvc.Value.ToInt == 1;
        }

        public Expr MaybeBuildPredicate(GPUVerifier verifier, Expr e)
        {
            var msc = this as ModStrideConstraint;
            if (msc != null && !msc.IsBottom())
            {
                Expr modEqExpr = Expr.Eq(verifier.IntRep.MakeModPow2(e, msc.Mod), verifier.IntRep.MakeModPow2(msc.ModEq, msc.Mod));
                return modEqExpr;
            }

            return null;
        }

        private static StrideConstraint BuildAddStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc)
        {
            if (lhsc is EqStrideConstraint && rhsc is EqStrideConstraint)
                return new EqStrideConstraint(e);

            if (lhsc is EqStrideConstraint && rhsc is ModStrideConstraint)
                return BuildAddStrideConstraint(verifier, e, rhsc, lhsc);

            if (lhsc is ModStrideConstraint && rhsc is EqStrideConstraint)
            {
                var lhsmc = (ModStrideConstraint)lhsc;
                var rhsec = (EqStrideConstraint)rhsc;

                return new ModStrideConstraint(lhsmc.Mod, verifier.IntRep.MakeAdd(lhsmc.ModEq, rhsec.Eq));
            }

            if (lhsc is ModStrideConstraint && rhsc is ModStrideConstraint)
            {
                var lhsmc = (ModStrideConstraint)lhsc;
                var rhsmc = (ModStrideConstraint)rhsc;

                if (lhsmc.Mod == rhsmc.Mod)
                    return new ModStrideConstraint(lhsmc.Mod, verifier.IntRep.MakeAdd(lhsmc.ModEq, rhsmc.ModEq));
            }

            return Bottom(verifier, e);
        }

        private static StrideConstraint BuildMulStrideConstraint(GPUVerifier verifier, Expr e, StrideConstraint lhsc, StrideConstraint rhsc)
        {
            if (lhsc is EqStrideConstraint && rhsc is EqStrideConstraint)
                return new EqStrideConstraint(e);

            if (lhsc is EqStrideConstraint && rhsc is ModStrideConstraint)
                return BuildMulStrideConstraint(verifier, e, rhsc, lhsc);

            if (lhsc is ModStrideConstraint && rhsc is EqStrideConstraint)
            {
                var lhsmc = (ModStrideConstraint)lhsc;
                var rhsec = (EqStrideConstraint)rhsc;

                return new ModStrideConstraint(
                    verifier.IntRep.MakeMul(lhsmc.Mod, rhsec.Eq),
                    verifier.IntRep.MakeMul(lhsmc.ModEq, rhsec.Eq));
            }

            return Bottom(verifier, e);
        }

        public static StrideConstraint FromExpr(GPUVerifier verifier, Implementation impl, Expr e)
        {
            if (e is LiteralExpr)
                return new EqStrideConstraint(e);

            var ee = e as BvExtractExpr;
            if (ee != null)
            {
                var sc = FromExpr(verifier, impl, ee.Bitvector);
                var modsc = sc as ModStrideConstraint;
                if (modsc != null)
                {
                    modsc = new ModStrideConstraint(
                        new BvExtractExpr(Token.NoToken, modsc.Mod, ee.End, ee.Start),
                        new BvExtractExpr(Token.NoToken, modsc.ModEq, ee.End, ee.Start));
                    modsc.Mod.Type = e.Type;
                    modsc.ModEq.Type = e.Type;
                    return modsc;
                }
                else
                {
                    return sc;
                }
            }

            var ie = e as IdentifierExpr;
            if (ie != null)
            {
                if (GPUVerifier.IsConstantInCurrentRegion(ie))
                    return new EqStrideConstraint(e);

                var rsa = verifier.ReducedStrengthAnalysesRegion[impl];
                var sc = rsa.GetStrideConstraint(ie.Decl.Name);
                if (sc == null)
                    return Bottom(verifier, e);
                return sc;
            }

            Expr lhs, rhs;

            if (verifier.IntRep.IsAdd(e, out lhs, out rhs))
            {
                var lhsc = FromExpr(verifier, impl, lhs);
                var rhsc = FromExpr(verifier, impl, rhs);
                return BuildAddStrideConstraint(verifier, e, lhsc, rhsc);
            }

            if (verifier.IntRep.IsMul(e, out lhs, out rhs))
            {
                var lhsc = FromExpr(verifier, impl, lhs);
                var rhsc = FromExpr(verifier, impl, rhs);
                return BuildMulStrideConstraint(verifier, e, lhsc, rhsc);
            }

            return Bottom(verifier, e);
        }
    }

    public class EqStrideConstraint : StrideConstraint
    {
        public EqStrideConstraint(Expr eq)
        {
            Eq = eq;
        }

        public Expr Eq { get; private set; }
    }

    public class ModStrideConstraint : StrideConstraint
    {
        public ModStrideConstraint(Expr mod, Expr modEq)
        {
            this.Mod = mod;
            this.ModEq = modEq;
        }

        public Expr Mod { get; private set; }

        public Expr ModEq { get; private set; }
    }
}
