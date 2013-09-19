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
using System.Diagnostics;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace GPUVerify.InvariantGenerationRules
{
    class PowerOfTwoInvariantGenerator : InvariantGenerationRule
    {

        public PowerOfTwoInvariantGenerator(GPUVerifier verifier)
            : base(verifier)
        {

        }

        public override void GenerateCandidates(Implementation Impl, IRegion region)
        {
            HashSet<Variable> modset = LoopInvariantGenerator.GetModifiedVariables(region);

            foreach (Variable v in Impl.LocVars)
            {
                string basicName = GPUVerifier.StripThreadIdentifier(v.Name);
                if (verifier.uniformityAnalyser.IsUniform(Impl.Name, basicName))
                {
                    if (verifier.mayBePowerOfTwoAnalyser.MayBePowerOfTwo(Impl.Name, basicName))
                    {
                        if (verifier.ContainsNamedVariable(modset, basicName))
                        {
                            verifier.AddCandidateInvariant(region, MakePowerOfTwoExpr(v), "pow2 disjunction", InferenceStages.BASIC_CANDIDATE_STAGE);
                            for (int i = (1 << 15); i > 0; i >>= 1)
                            {
                                verifier.AddCandidateInvariant(region,
                                    verifier.IntRep.MakeSlt(
                                    new IdentifierExpr(v.tok, v),
                                    verifier.IntRep.GetLiteral(i, 32)),
                                    "pow2 less than " + i, InferenceStages.BASIC_CANDIDATE_STAGE);
                            }
                            verifier.AddCandidateInvariant(region,
                                Expr.Neq(new IdentifierExpr(v.tok, v),
                                verifier.IntRep.GetLiteral(0, 32)),
                                "pow2 not zero", InferenceStages.BASIC_CANDIDATE_STAGE);
                        }
                    }
                }
            }

            // Relational Power Of Two
            var incs = modset.Where(v => verifier.relationalPowerOfTwoAnalyser.IsInc(Impl.Name, v.Name));
            var decs = modset.Where(v => verifier.relationalPowerOfTwoAnalyser.IsDec(Impl.Name, v.Name));
            if (incs.ToList().Count() == 1 && decs.ToList().Count() == 1)
            {
                var inc = incs.Single();
                var dec = decs.Single();
                for (int i = (1 << 15); i > 0; i >>= 1)
                {
                    var mulInv = Expr.Eq(verifier.IntRep.MakeMul(new IdentifierExpr(inc.tok, inc), new IdentifierExpr(dec.tok, dec)), verifier.IntRep.GetLiteral(i, 32));
                    verifier.AddCandidateInvariant(region, mulInv, "relational pow2", InferenceStages.BASIC_CANDIDATE_STAGE);
                    var disjInv = Expr.Or(
                      Expr.And(Expr.Eq(new IdentifierExpr(dec.tok, dec), verifier.IntRep.GetLiteral(0, 32)),
                               Expr.Eq(new IdentifierExpr(inc.tok, inc), verifier.IntRep.GetLiteral(2*i, 32))),
                      mulInv);
                    verifier.AddCandidateInvariant(region, disjInv, "relational pow2", InferenceStages.BASIC_CANDIDATE_STAGE);
                }
            }
        }

        private Expr MakePowerOfTwoExpr(Variable v)
        {
            Expr result = null;
            for (int i = 1 << 15; i > 0; i >>= 1)
            {
                Expr eq = Expr.Eq(new IdentifierExpr(v.tok, v),
                  verifier.IntRep.GetLiteral(i, 32));
                result = (result == null ? eq : Expr.Or(eq, result));
            }

            return Expr.Or(Expr.Eq(new IdentifierExpr(v.tok, v),
              verifier.IntRep.GetLiteral(0, 32)), result);
        }

    }
}
