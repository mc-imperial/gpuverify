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
    using System.Numerics;
    using System.Text.RegularExpressions;
    using Microsoft.Boogie;

    public class ExpressionSimplifier
    {
        public static void Simplify(Program program, IntegerRepresentation intRep)
        {
            var exprSimplify = new Simplifier(intRep);
            exprSimplify.Visit(program);
        }

        private class Simplifier : StandardVisitor
        {
            private static readonly Regex AndPattern = new Regex(@"^BV\d*_AND$");
            private static readonly Regex OrPattern = new Regex(@"^BV\d*_OR$");
            private static readonly Regex XorPattern = new Regex(@"^BV\d*_XOR$");
            private static readonly Regex ZextPattern = new Regex(@"^BV\d*_ZEXT\d*$");

            private IntegerRepresentation intRep;

            public Simplifier(IntegerRepresentation intRep)
            {
                this.intRep = intRep;
            }

            public override Expr VisitNAryExpr(NAryExpr node)
            {
                // Simplify the children.
                node = (NAryExpr)base.VisitNAryExpr(node);

                // Simplify constant Boolean and bitvector expressions that
                // occur in assertions when not all arrays are being checked
                // for races. Bitvector expressions are ignored if they were
                // modelled as mathematical integers.
                if (node.Fun is BinaryOperator)
                {
                    var binOp = node.Fun as BinaryOperator;
                    if (binOp.Op == BinaryOperator.Opcode.Imp)
                    {
                        if (node.Args[0] is LiteralExpr)
                        {
                            return node.Args[0] == Expr.False ? Expr.True : node.Args[1];
                        }
                        else if (node.Args[1] is LiteralExpr)
                        {
                            return node.Args[1] == Expr.True ? Expr.True : Expr.Not(node.Args[0]);
                        }
                    }
                    else if (binOp.Op == BinaryOperator.Opcode.Neq)
                    {
                        if (node.Args[0] is LiteralExpr && node.Args[1] is LiteralExpr)
                        {
                            return node.Args[0].Equals(node.Args[1]) ? Expr.False : Expr.True;
                        }
                    }
                    else if (binOp.Op == BinaryOperator.Opcode.Eq)
                    {
                        if (node.Args[0] is LiteralExpr && node.Args[1] is LiteralExpr)
                        {
                            return node.Args[0].Equals(node.Args[1]) ? Expr.True : Expr.False;
                        }
                    }
                }
                else if (ZextPattern.Match(node.Fun.FunctionName).Success && node.Type.IsBv)
                {
                    if (node.Args[0] is LiteralExpr)
                    {
                        var newVal = GetVal(node.Args[0]);
                        return intRep.GetLiteral(newVal, node.Type);
                    }
                }
                else if (XorPattern.Match(node.Fun.FunctionName).Success && node.Type.IsBv)
                {
                    if (node.Args[0] is LiteralExpr && node.Args[1] is LiteralExpr)
                    {
                        var newVal = GetVal(node.Args[0]) ^ GetVal(node.Args[1]);
                        return intRep.GetLiteral(newVal, node.Type);
                    }
                    else if (node.Args[0] is LiteralExpr && GetVal(node.Args[0]) == 0)
                    {
                        return node.Args[1];
                    }
                    else if (node.Args[1] is LiteralExpr && GetVal(node.Args[1]) == 0)
                    {
                        return node.Args[0];
                    }
                }
                else if (AndPattern.Match(node.Fun.FunctionName).Success && node.Type.IsBv)
                {
                    if (node.Args[0] is LiteralExpr && node.Args[1] is LiteralExpr)
                    {
                        var newVal = GetVal(node.Args[0]) & GetVal(node.Args[1]);
                        return intRep.GetLiteral(newVal, node.Type);
                    }
                    else if (node.Args[0] is LiteralExpr && GetVal(node.Args[0]) == 0)
                    {
                        return node.Args[0];
                    }
                    else if (node.Args[1] is LiteralExpr && GetVal(node.Args[1]) == 0)
                    {
                        return node.Args[1];
                    }
                }
                else if (OrPattern.Match(node.Fun.FunctionName).Success && node.Type.IsBv)
                {
                    if (node.Args[0] is LiteralExpr && node.Args[1] is LiteralExpr)
                    {
                        var newVal = GetVal(node.Args[0]) | GetVal(node.Args[1]);
                        return intRep.GetLiteral(newVal, node.Type);
                    }
                    else if (node.Args[0] is LiteralExpr && GetVal(node.Args[0]) == 0)
                    {
                        return node.Args[1];
                    }
                    else if (node.Args[1] is LiteralExpr && GetVal(node.Args[1]) == 0)
                    {
                        return node.Args[0];
                    }
                }
                else if (node.Fun is IfThenElse)
                {
                    if (node.Args[0] is LiteralExpr)
                    {
                        return node.Args[0] == Expr.True ? node.Args[1] : node.Args[2];
                    }
                }

                return node;
            }

            private static BigInteger GetVal(Expr node)
            {
                var litExpr = (LiteralExpr)node;
                if (litExpr.isBvConst)
                {
                    return litExpr.asBvConst.Value.ToBigInteger;
                }
                else if (litExpr.isBigNum)
                {
                    return litExpr.asBigNum.ToBigInteger;
                }
                else
                {
                    throw new NotSupportedException("Type not supported");
                }
            }
        }
    }
}
