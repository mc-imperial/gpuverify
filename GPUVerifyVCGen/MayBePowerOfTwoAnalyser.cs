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

namespace GPUVerify
{
    class MayBePowerOfTwoAnalyser
    {
        private GPUVerifier verifier;

        private Dictionary<string, Dictionary<string, bool>> mayBePowerOfTwoInfo;

        public MayBePowerOfTwoAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            mayBePowerOfTwoInfo = new Dictionary<string, Dictionary<string, bool>>();
        }

        internal void Analyse()
        {
            foreach (Declaration D in verifier.Program.TopLevelDeclarations)
            {
                if (D is Implementation)
                {
                    Implementation Impl = D as Implementation;
                    mayBePowerOfTwoInfo.Add(Impl.Name, new Dictionary<string, bool>());

                    SetNotPowerOfTwo(Impl.Name, GPUVerifier._X.Name);
                    SetNotPowerOfTwo(Impl.Name, GPUVerifier._Y.Name);
                    SetNotPowerOfTwo(Impl.Name, GPUVerifier._Z.Name);

                    foreach (Variable v in Impl.LocVars)
                    {
                        SetNotPowerOfTwo(Impl.Name, v.Name);
                    }

                    foreach (Variable v in Impl.InParams)
                    {
                        SetNotPowerOfTwo(Impl.Name, v.Name);
                    }

                    foreach (Variable v in Impl.OutParams)
                    {
                        SetNotPowerOfTwo(Impl.Name, v.Name);
                    }

                    // Fixpoint not required - this is just syntactic
                    Analyse(Impl);

                }
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowMayBePowerOfTwoAnalysis)
            {
                dump();
            }
        }

        private void SetNotPowerOfTwo(string p, string v)
        {
            mayBePowerOfTwoInfo[p][v] = false;
        }

        private void Analyse(Implementation Impl)
        {
            Analyse(Impl, verifier.RootRegion(Impl));
        }

        private void Analyse(Implementation impl, IRegion region)
        {
            foreach (Cmd c in region.Cmds())
            {
                if (c is AssignCmd)
                {
                    AssignCmd assign = c as AssignCmd;

                    for (int i = 0; i != assign.Lhss.Count; i++)
                    {
                        if (assign.Lhss[i] is SimpleAssignLhs)
                        {
                            Variable v = (assign.Lhss[i] as SimpleAssignLhs).AssignedVariable.Decl;
                            if (mayBePowerOfTwoInfo[impl.Name].ContainsKey(v.Name))
                            {
                                if (isPowerOfTwoOperation(v, assign.Rhss[i]))
                                {
                                    mayBePowerOfTwoInfo[impl.Name][v.Name] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        private bool isPowerOfTwoOperation(Variable v, Expr expr)
        {

            if (!(
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(8)) ||
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(16)) ||
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(32))
                ))
            {
                return false;
            }

            Expr lhs, rhs;

            if (IntegerRepresentationHelper.IsFun(expr, "MUL", out lhs, out rhs)) {
                return
                   (
                    (IsVariable(lhs, v) || IsVariable(rhs, v)) &&
                    (IsConstant(lhs, 2) || IsConstant(rhs, 2))
                    );
            }

            if (IntegerRepresentationHelper.IsFun(expr, "DIV", out lhs, out rhs) ||
                IntegerRepresentationHelper.IsFun(expr, "SDIV", out lhs, out rhs)) {
                return
                   (
                    IsVariable(lhs, v) && IsConstant(rhs, 2)
                    );
            }

            if (IntegerRepresentationHelper.IsFun(expr, "SHL", out lhs, out rhs)) {
                return
                   (
                    IsVariable(lhs, v) && IsConstant(rhs, 1)
                    );
            }

            if (IntegerRepresentationHelper.IsFun(expr, "ASHR", out lhs, out rhs)) {
                return
                   (
                    IsVariable(lhs, v) && IsConstant(rhs, 1)
                    );
            }

            if (IntegerRepresentationHelper.IsFun(expr, "LSHR", out lhs, out rhs)) {
              return
                 (
                  IsVariable(lhs, v) && IsConstant(rhs, 1)
                  );
            }

            return false;
        }

        private bool IsConstant(Expr expr, int x)
        {
            if (!(expr is LiteralExpr))
            {
                return false;
            }

            LiteralExpr lit = expr as LiteralExpr;

            if (lit.Val is BvConst)
            {
                return (lit.Val as BvConst).Value.ToInt == x;
            }

            if (lit.Val is Microsoft.Basetypes.BigNum)
            {
                return ((Microsoft.Basetypes.BigNum)lit.Val).ToInt == x;
            }

            return false;

        }

        private bool IsVariable(Expr expr, Variable v)
        {
            return expr is IdentifierExpr && ((expr as IdentifierExpr).Decl.Name.Equals(v.Name));
        }

        private void dump()
        {
            foreach (string p in mayBePowerOfTwoInfo.Keys)
            {
                Console.WriteLine("Procedure " + p);
                foreach (string v in mayBePowerOfTwoInfo[p].Keys)
                {
                    Console.WriteLine("  " + v + ": " +
                        (mayBePowerOfTwoInfo[p][v] ? "may be power of two" : "likely not power of two"));
                }
            }

        }


        internal bool MayBePowerOfTwo(string p, string v)
        {
            if (!mayBePowerOfTwoInfo.ContainsKey(p))
            {
                return false;
            }

            if (!mayBePowerOfTwoInfo[p].ContainsKey(v))
            {
                return false;
            }

            return mayBePowerOfTwoInfo[p][v];
        }
    }
}
