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

    internal class MayBePowerOfTwoAnalyser
    {
        private GPUVerifier verifier;

        // mayBePowerOfTwoInfo: impl -> var -> ispow2
        private Dictionary<string, Dictionary<string, bool>> mayBePowerOfTwoInfo;

        public MayBePowerOfTwoAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            mayBePowerOfTwoInfo = new Dictionary<string, Dictionary<string, bool>>();
        }

        public void Analyse()
        {
            foreach (Declaration decl in verifier.Program.TopLevelDeclarations)
            {
                if (decl is Implementation)
                {
                    Implementation impl = decl as Implementation;
                    mayBePowerOfTwoInfo.Add(impl.Name, new Dictionary<string, bool>());

                    SetNotPowerOfTwo(impl.Name, GPUVerifier._X.Name);
                    SetNotPowerOfTwo(impl.Name, GPUVerifier._Y.Name);
                    SetNotPowerOfTwo(impl.Name, GPUVerifier._Z.Name);

                    foreach (Variable v in impl.LocVars)
                    {
                        SetNotPowerOfTwo(impl.Name, v.Name);
                    }

                    foreach (Variable v in impl.InParams)
                    {
                        SetNotPowerOfTwo(impl.Name, v.Name);
                    }

                    foreach (Variable v in impl.OutParams)
                    {
                        SetNotPowerOfTwo(impl.Name, v.Name);
                    }

                    // Fixpoint not required - this is just syntactic
                    Analyse(impl);
                }
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowMayBePowerOfTwoAnalysis)
                Dump();
        }

        public bool MayBePowerOfTwo(string p, string v)
        {
            if (!mayBePowerOfTwoInfo.ContainsKey(p))
                return false;

            if (!mayBePowerOfTwoInfo[p].ContainsKey(v))
                return false;

            return mayBePowerOfTwoInfo[p][v];
        }

        private void SetNotPowerOfTwo(string p, string v)
        {
            mayBePowerOfTwoInfo[p][v] = false;
        }

        private void Analyse(Implementation impl)
        {
            Analyse(impl, verifier.RootRegion(impl));
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
                            Variable lhs = (assign.Lhss[i] as SimpleAssignLhs).AssignedVariable.Decl;
                            if (VariableHasPowerOfTwoType(lhs) &&
                                mayBePowerOfTwoInfo[impl.Name].ContainsKey(lhs.Name))
                            {
                                Variable rhs = GetPowerOfTwoRhsVariable(assign.Rhss[i]);
                                if (rhs != null)
                                {
                                    mayBePowerOfTwoInfo[impl.Name][lhs.Name] = true;
                                    mayBePowerOfTwoInfo[impl.Name][rhs.Name] = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        private bool VariableHasPowerOfTwoType(Variable v)
        {
            return
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(8)) ||
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(16)) ||
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(32)) ||
                v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(64));
        }

        private Variable GetPowerOfTwoRhsVariable(Expr expr)
        {
            Expr lhs, rhs;

            if (IntegerRepresentationHelper.IsFun(expr, "MUL", out lhs, out rhs))
            {
                if (IsVariable(lhs) && IsConstant(rhs, 2))
                    return GetVariable(lhs);
                else if (IsConstant(lhs, 2) && IsVariable(rhs))
                    return GetVariable(rhs);
                else
                    return null;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "DIV", out lhs, out rhs) ||
                IntegerRepresentationHelper.IsFun(expr, "SDIV", out lhs, out rhs))
            {
                if (IsVariable(lhs) && IsConstant(rhs, 2))
                    return GetVariable(lhs);
                else
                    return null;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "SHL", out lhs, out rhs) ||
                IntegerRepresentationHelper.IsFun(expr, "ASHR", out lhs, out rhs) ||
                IntegerRepresentationHelper.IsFun(expr, "LSHR", out lhs, out rhs))
            {
                if (IsVariable(lhs) && IsConstant(rhs))
                    return GetVariable(lhs);
                else
                    return null;
            }

            return null;
        }

        private bool IsConstant(Expr expr)
        {
            if (!(expr is LiteralExpr))
            {
                return false;
            }

            LiteralExpr lit = expr as LiteralExpr;

            return (lit.Val is BvConst) ||
                    (lit.Val is Microsoft.Basetypes.BigNum);
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
                if (((BvConst)lit.Val).Value.InInt32)
                {
                    return ((BvConst)lit.Val).Value.ToIntSafe == x;
                }
            }

            if (lit.Val is Microsoft.Basetypes.BigNum)
            {
                if (((Microsoft.Basetypes.BigNum)lit.Val).InInt32)
                {
                    return ((Microsoft.Basetypes.BigNum)lit.Val).ToInt == x;
                }
            }

            return false;
        }

        private bool IsVariable(Expr expr)
        {
            return expr is IdentifierExpr;
        }

        private Variable GetVariable(Expr expr)
        {
            return (expr as IdentifierExpr).Decl;
        }

        private void Dump()
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
    }
}
