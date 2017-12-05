//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify.InvariantGenerationRules
{
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.Boogie;

    public class PowerOfTwoInvariantGenerator : InvariantGenerationRule
    {
        public PowerOfTwoInvariantGenerator(GPUVerifier verifier)
            : base(verifier)
        {
        }

        public override void GenerateCandidates(Implementation impl, IRegion region)
        {
            HashSet<Variable> modset = region.GetModifiedVariables();

            foreach (Variable v in impl.LocVars)
            {
                string basicName = Utilities.StripThreadIdentifier(v.Name);
                if (Verifier.MayBePowerOfTwoAnalyser.MayBePowerOfTwo(impl.Name, basicName))
                {
                    if (Verifier.ContainsNamedVariable(modset, basicName))
                    {
                        var bitwiseInv = Expr.Or(
                            Expr.Eq(new IdentifierExpr(v.tok, v), Verifier.Zero(32)),
                            Expr.Eq(
                                Verifier.IntRep.MakeAnd(
                                    new IdentifierExpr(v.tok, v),
                                    Verifier.IntRep.MakeSub(
                                        new IdentifierExpr(v.tok, v), Verifier.IntRep.GetLiteral(1, 32))),
                                Verifier.Zero(32)));
                        Verifier.AddCandidateInvariant(region, bitwiseInv, "pow2");

                        Verifier.AddCandidateInvariant(
                            region,
                            Expr.Neq(new IdentifierExpr(v.tok, v), Verifier.Zero(32)),
                            "pow2NotZero");
                    }
                }
            }

            // Relational Power Of Two
            var incs = modset.Where(v => Verifier.RelationalPowerOfTwoAnalyser.IsInc(impl.Name, v.Name));
            var decs = modset.Where(v => Verifier.RelationalPowerOfTwoAnalyser.IsDec(impl.Name, v.Name));
            if (incs.ToList().Count() == 1 && decs.ToList().Count() == 1)
            {
                var inc = incs.Single();
                var dec = decs.Single();
                for (int i = 1 << 15; i > 0; i >>= 1)
                {
                    var mulInv = Expr.Eq(Verifier.IntRep.MakeMul(new IdentifierExpr(inc.tok, inc), new IdentifierExpr(dec.tok, dec)), Verifier.IntRep.GetLiteral(i, 32));
                    Verifier.AddCandidateInvariant(region, mulInv, "relationalPow2");
                    var disjInv = Expr.Or(
                      Expr.And(
                          Expr.Eq(new IdentifierExpr(dec.tok, dec), Verifier.Zero(32)),
                          Expr.Eq(new IdentifierExpr(inc.tok, inc), Verifier.IntRep.GetLiteral(2 * i, 32))),
                      mulInv);
                    Verifier.AddCandidateInvariant(region, disjInv, "relationalPow2");
                }
            }
        }
    }
}
