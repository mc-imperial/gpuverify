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
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Numerics;
    using System.Text.RegularExpressions;
    using Microsoft.Basetypes;
    using Microsoft.Boogie;

    public interface IntegerRepresentation
    {
        Type GetIntType(int width);

        LiteralExpr GetZero(Type resultType);

        LiteralExpr GetLiteral(int value, Type resultType);

        LiteralExpr GetLiteral(BigInteger value, Type resultType);

        Expr MakeSub(Expr lhs, Expr rhs);

        Expr MakeAnd(Expr lhs, Expr rhs);

        Expr MakeUlt(Expr lhs, Expr rhs);

        Expr MakeUle(Expr lhs, Expr rhs);

        Expr MakeUgt(Expr lhs, Expr rhs);

        Expr MakeUge(Expr lhs, Expr rhs);

        Expr MakeSlt(Expr lhs, Expr rhs);

        Expr MakeSle(Expr lhs, Expr rhs);

        Expr MakeSgt(Expr lhs, Expr rhs);

        Expr MakeSge(Expr lhs, Expr rhs);

        Expr MakeAdd(Expr lhs, Expr rhs);

        Expr MakeMul(Expr lhs, Expr rhs);

        Expr MakeDiv(Expr lhs, Expr rhs);

        Expr MakeModPow2(Expr lhs, Expr rhs);

        Expr MakeZext(Expr expr, Type resultType);

        bool IsAdd(Expr e, out Expr lhs, out Expr rhs);

        bool IsMul(Expr e, out Expr lhs, out Expr rhs);
    }

    public class IntegerRepresentationHelper
    {
        public static bool IsFun(Expr e, string mnemonic, out Expr lhs, out Expr rhs)
        {
            lhs = rhs = null;

            var ne = e as NAryExpr;
            if (ne == null)
            {
                return false;
            }

            var fc = ne.Fun as FunctionCall;
            if (fc == null)
            {
                return false;
            }

            if (!Regex.IsMatch(fc.FunctionName, "BV[0-9]+_" + mnemonic))
            {
                return false;
            }

            lhs = ne.Args[0];
            rhs = ne.Args[1];
            return true;
        }
    }

    public class BVIntegerRepresentation : IntegerRepresentation
    {
        private GPUVerifier verifier;

        public BVIntegerRepresentation(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public Type GetIntType(int width)
        {
            return Type.GetBvType(width);
        }

        public LiteralExpr GetZero(Type resultType)
        {
            return GetLiteral(0, resultType);
        }

        public LiteralExpr GetLiteral(int value, Type resultType)
        {
            Debug.Assert(resultType.IsBv);
            return new LiteralExpr(Token.NoToken, BigNum.FromInt(value), resultType.BvBits);
        }

        public LiteralExpr GetLiteral(BigInteger value, Type resultType)
        {
            Debug.Assert(resultType.IsBv);
            var v = value;

            if (v < 0)
            {
                v += BigInteger.Pow(2, resultType.BvBits);
            }

            return new LiteralExpr(Token.NoToken, BigNum.FromBigInt(v), resultType.BvBits);
        }

        private Expr MakeBitVectorBinaryBoolean(string suffix, string smtName, Expr lhs, Expr rhs)
        {
            return MakeBVFunctionCall("BV" + lhs.Type.BvBits + "_" + suffix, smtName, Type.Bool, lhs, rhs);
        }

        private Expr MakeBitVectorUnaryBitVector(string suffix, string smtName, Expr expr, Type resultType)
        {
            return MakeBVFunctionCall("BV" + expr.Type.BvBits + "_" + suffix, smtName, resultType, expr);
        }

        private Expr MakeBitVectorBinaryBitVector(string suffix, string smtName, Expr lhs, Expr rhs)
        {
            return MakeBVFunctionCall("BV" + lhs.Type.BvBits + "_" + suffix, smtName, lhs.Type, lhs, rhs);
        }

        private Expr MakeBVFunctionCall(string functionName, string smtName, Type resultType, params Expr[] args)
        {
            Function f = verifier.GetOrCreateBVFunction(functionName, smtName, resultType, args.Select(a => a.Type).ToArray());
            var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr>(args));
            e.Type = resultType;
            return e;
        }

        public Expr MakeSub(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBitVector("SUB", "bvsub", lhs, rhs);
        }

        public Expr MakeAnd(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBitVector("AND", "bvand", lhs, rhs);
        }

        public Expr MakeUge(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("UGE", "bvuge", lhs, rhs);
        }

        public Expr MakeUlt(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("ULT", "bvult", lhs, rhs);
        }

        public Expr MakeUle(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("ULE", "bvule", lhs, rhs);
        }

        public Expr MakeUgt(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("UGT", "bvugt", lhs, rhs);
        }

        public Expr MakeSge(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("SGE", "bvsge", lhs, rhs);
        }

        public Expr MakeSlt(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("SLT", "bvslt", lhs, rhs);
        }

        public Expr MakeSle(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("SLE", "bvsle", lhs, rhs);
        }

        public Expr MakeSgt(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBoolean("SGT", "bvsgt", lhs, rhs);
        }

        public Expr MakeAdd(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBitVector("ADD", "bvadd", lhs, rhs);
        }

        public Expr MakeMul(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBitVector("MUL", "bvmul", lhs, rhs);
        }

        public Expr MakeDiv(Expr lhs, Expr rhs)
        {
            return MakeBitVectorBinaryBitVector("DIV", "bvsdiv", lhs, rhs);
        }

        public Expr MakeModPow2(Expr lhs, Expr rhs)
        {
            var bvType = rhs.Type as BvType;
            return MakeAnd(MakeSub(rhs, GetLiteral(1, bvType)), lhs);
        }

        public Expr MakeZext(Expr expr, Type resultType)
        {
            Debug.Assert(resultType.IsBv);
            if (expr.Type.BvBits == resultType.BvBits)
                return expr;
            else
                return MakeBitVectorUnaryBitVector("ZEXT" + resultType.BvBits, "zero_extend " + (resultType.BvBits - expr.Type.BvBits), expr, resultType);
        }

        public bool IsAdd(Expr e, out Expr lhs, out Expr rhs)
        {
            return IntegerRepresentationHelper.IsFun(e, "ADD", out lhs, out rhs);
        }

        public bool IsMul(Expr e, out Expr lhs, out Expr rhs)
        {
            return IntegerRepresentationHelper.IsFun(e, "MUL", out lhs, out rhs);
        }
    }

    public class MathIntegerRepresentation : IntegerRepresentation
    {
        private GPUVerifier verifier;

        public MathIntegerRepresentation(GPUVerifier verifier)
        {
            this.verifier = verifier;
        }

        public Type GetIntType(int width)
        {
            return Type.Int;
        }

        public LiteralExpr GetZero(Type resultType)
        {
            return GetLiteral(0, resultType);
        }

        public LiteralExpr GetLiteral(int value, Type resultType)
        {
            Debug.Assert(resultType.IsInt);
            return new LiteralExpr(Token.NoToken, BigNum.FromInt(value));
        }

        public LiteralExpr GetLiteral(BigInteger value, Type resultType)
        {
            Debug.Assert(resultType.IsInt);
            return new LiteralExpr(Token.NoToken, BigNum.FromBigInt(value));
        }

        private Expr MakeIntBinaryInt(string suffix, BinaryOperator.Opcode infixOp, Expr lhs, Expr rhs)
        {
            return MakeIntFunctionCall("BV32_" + suffix, infixOp, lhs.Type, lhs, rhs);
        }

        private Expr MakeIntBinaryIntUF(string suffix, Expr lhs, Expr rhs)
        {
            return MakeIntUFFunctionCall("BV32_" + suffix, lhs.Type, lhs, rhs);
        }

        private Expr MakeIntBinaryBool(string suffix, BinaryOperator.Opcode infixOp, Expr lhs, Expr rhs)
        {
            return MakeIntFunctionCall("BV32_" + suffix, infixOp, Type.Bool, lhs, rhs);
        }

        private Expr MakeIntFunctionCall(string functionName, BinaryOperator.Opcode infixOp, Type resultType, Expr lhs, Expr rhs)
        {
            Function f = verifier.GetOrCreateIntFunction(functionName, infixOp, resultType, lhs.Type, rhs.Type);
            var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr> { lhs, rhs });
            e.Type = resultType;
            return e;
        }

        private Expr MakeIntUFFunctionCall(string functionName, Type resultType, Expr lhs, Expr rhs)
        {
            Function f = verifier.GetOrCreateBinaryUF(functionName, resultType, lhs.Type, rhs.Type);
            var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr> { lhs, rhs });
            e.Type = resultType;
            return e;
        }

        public Expr MakeSub(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryInt("SUB", BinaryOperator.Opcode.Sub, lhs, rhs);
        }

        public Expr MakeAdd(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryInt("ADD", BinaryOperator.Opcode.Add, lhs, rhs);
        }

        public Expr MakeMul(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryInt("MUL", BinaryOperator.Opcode.Mul, lhs, rhs);
        }

        public Expr MakeDiv(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryInt("MUL", BinaryOperator.Opcode.Div, lhs, rhs);
        }

        public Expr MakeAnd(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryIntUF("AND", lhs, rhs);
        }

        public Expr MakeUgt(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("UGT", BinaryOperator.Opcode.Gt, lhs, rhs);
        }

        public Expr MakeUge(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("UGE", BinaryOperator.Opcode.Ge, lhs, rhs);
        }

        public Expr MakeUlt(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("ULT", BinaryOperator.Opcode.Lt, lhs, rhs);
        }

        public Expr MakeUle(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("ULE", BinaryOperator.Opcode.Le, lhs, rhs);
        }

        public Expr MakeSgt(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("SGT", BinaryOperator.Opcode.Gt, lhs, rhs);
        }

        public Expr MakeSge(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("SGE", BinaryOperator.Opcode.Ge, lhs, rhs);
        }

        public Expr MakeSlt(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("SLT", BinaryOperator.Opcode.Lt, lhs, rhs);
        }

        public Expr MakeSle(Expr lhs, Expr rhs)
        {
            return MakeIntBinaryBool("SLE", BinaryOperator.Opcode.Le, lhs, rhs);
        }

        public Expr MakeModPow2(Expr lhs, Expr rhs)
        {
            return Expr.Binary(BinaryOperator.Opcode.Mod, lhs, rhs);
        }

        public Expr MakeZext(Expr expr, Type resultType)
        {
            Debug.Assert(resultType.IsInt);
            return expr;
        }

        public bool IsAdd(Expr e, out Expr lhs, out Expr rhs)
        {
            return IntegerRepresentationHelper.IsFun(e, "ADD", out lhs, out rhs);
        }

        public bool IsMul(Expr e, out Expr lhs, out Expr rhs)
        {
            return IntegerRepresentationHelper.IsFun(e, "MUL", out lhs, out rhs);
        }
    }
}
