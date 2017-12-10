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
    using System.Diagnostics;
    using System.Text.RegularExpressions;
    using Microsoft.Boogie;

    public class RelationalPowerOfTwoAnalyser
    {
        private GPUVerifier verifier;
        private Dictionary<string, Dictionary<string, Kind>> mayBePowerOfTwoInfo;

        private enum Kind
        {
            No, Inc, Dec
        }

        public RelationalPowerOfTwoAnalyser(GPUVerifier verifier)
        {
            this.verifier = verifier;
            mayBePowerOfTwoInfo = new Dictionary<string, Dictionary<string, Kind>>();
        }

        public bool IsInc(string p, string v)
        {
            if (!mayBePowerOfTwoInfo.ContainsKey(p))
            {
                return false;
            }

            if (!mayBePowerOfTwoInfo[p].ContainsKey(v))
            {
                return false;
            }

            return mayBePowerOfTwoInfo[p][v] == Kind.Inc;
        }

        public bool IsDec(string p, string v)
        {
            if (!mayBePowerOfTwoInfo.ContainsKey(p))
            {
                return false;
            }

            if (!mayBePowerOfTwoInfo[p].ContainsKey(v))
            {
                return false;
            }

            return mayBePowerOfTwoInfo[p][v] == Kind.Dec;
        }

        public void Analyse()
        {
            foreach (Declaration decl in verifier.Program.TopLevelDeclarations)
            {
                if (decl is Implementation)
                {
                    Implementation impl = decl as Implementation;
                    mayBePowerOfTwoInfo.Add(impl.Name, new Dictionary<string, Kind>());

                    SetNotPowerOfTwo(impl.Name, verifier.IdX.Name);
                    SetNotPowerOfTwo(impl.Name, verifier.IdY.Name);
                    SetNotPowerOfTwo(impl.Name, verifier.IdZ.Name);

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
            {
                Dump();
            }
        }

        private void SetNotPowerOfTwo(string p, string v)
        {
            mayBePowerOfTwoInfo[p][v] = Kind.No;
        }

        private void Analyse(Implementation impl)
        {
            Analyse(impl, verifier.RootRegion(impl));
        }

        private bool IsTempVariable(Expr expr)
        {
            if (expr is IdentifierExpr)
            {
                IdentifierExpr iexpr = expr as IdentifierExpr;
                string name = iexpr.Name;
                Match match = Regex.Match(name, @"v[0-9]+");
                return match.Success;
            }

            return false;
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
                                Expr expr = assign.Rhss[i];
                                if (IsTempVariable(expr))
                                    expr = verifier.VarDefAnalysesRegion[impl].DefOfVariableName((expr as IdentifierExpr).Name);

                                switch (IsPowerOfTwoOperation(v, expr))
                                {
                                    case Kind.No:
                                        break;
                                    case Kind.Inc:
                                        mayBePowerOfTwoInfo[impl.Name][v.Name] = Kind.Inc;
                                        break;
                                    case Kind.Dec:
                                        mayBePowerOfTwoInfo[impl.Name][v.Name] = Kind.Dec;
                                        break;
                                    default:
                                        Debug.Assert(false);
                                        break;
                                }
                            }
                        }
                    }
                }
            }
        }

        private Kind IsPowerOfTwoOperation(Variable v, Expr expr)
        {
            if (!(v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(8))
                || v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(16))
                || v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(32))
                || v.TypedIdent.Type.Equals(verifier.IntRep.GetIntType(64))))
            {
                return Kind.No;
            }

            Expr lhs, rhs;

            if (IntegerRepresentationHelper.IsFun(expr, "MUL", out lhs, out rhs))
            {
                if ((IsVariable(lhs, v) || IsVariable(rhs, v)) && (IsConstant(lhs, 2) || IsConstant(rhs, 2)))
                    return Kind.Inc;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "DIV", out lhs, out rhs)
                || IntegerRepresentationHelper.IsFun(expr, "SDIV", out lhs, out rhs))
            {
                if (IsVariable(lhs, v) && IsConstant(rhs, 2))
                    return Kind.Dec;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "SHL", out lhs, out rhs))
            {
                if (IsVariable(lhs, v) && IsConstant(rhs, 1))
                    return Kind.Inc;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "ASHR", out lhs, out rhs))
            {
                if (IsVariable(lhs, v) && IsConstant(rhs, 1))
                    return Kind.Dec;
            }

            if (IntegerRepresentationHelper.IsFun(expr, "LSHR", out lhs, out rhs))
            {
                if (IsVariable(lhs, v) && IsConstant(rhs, 1))
                    return Kind.Dec;
            }

            return Kind.No;
        }

        private bool IsConstant(Expr expr, int x)
        {
            if (!(expr is LiteralExpr))
                return false;

            LiteralExpr lit = expr as LiteralExpr;

            if (lit.Val is BvConst)
                return (lit.Val as BvConst).Value.ToInt == x;

            if (lit.Val is Microsoft.Basetypes.BigNum)
                return ((Microsoft.Basetypes.BigNum)lit.Val).ToInt == x;

            return false;
        }

        private bool IsVariable(Expr expr, Variable v)
        {
            return expr is IdentifierExpr && (expr as IdentifierExpr).Decl.Name.Equals(v.Name);
        }

        private void Dump()
        {
            foreach (string p in mayBePowerOfTwoInfo.Keys)
            {
                Console.WriteLine("Procedure " + p);
                foreach (string v in mayBePowerOfTwoInfo[p].Keys)
                {
                    Console.WriteLine("  " + v + ": " +
                        (mayBePowerOfTwoInfo[p][v] == Kind.No ? "likely not power of two" :
                         mayBePowerOfTwoInfo[p][v] == Kind.Inc ? "maybe incrementing power of two" :
                         mayBePowerOfTwoInfo[p][v] == Kind.Dec ? "maybe decrementing power of two" :
                         string.Empty));
                }
            }
        }
    }
}
