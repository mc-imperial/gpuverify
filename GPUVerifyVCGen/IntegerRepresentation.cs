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
using System.Text.RegularExpressions;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace GPUVerify
{
  interface IntegerRepresentation
  {
    Microsoft.Boogie.Type GetIntType(int Width);
    LiteralExpr GetLiteral(int Value, int Width);

    Expr MakeSub(Expr lhs, Expr rhs);
    Expr MakeAnd(Expr lhs, Expr rhs);
    Expr MakeSlt(Expr lhs, Expr rhs);
    Expr MakeSle(Expr lhs, Expr rhs);
    Expr MakeSgt(Expr lhs, Expr rhs);
    Expr MakeSge(Expr lhs, Expr rhs);
    Expr MakeAdd(Expr lhs, Expr rhs);
    Expr MakeMul(Expr lhs, Expr rhs);
    Expr MakeDiv(Expr lhs, Expr rhs);
    Expr MakeModPow2(Expr lhs, Expr rhs);
    Expr MakeZext(Expr expr, Microsoft.Boogie.Type resultType);
    bool IsAdd(Expr e, out Expr lhs, out Expr rhs);
    bool IsMul(Expr e, out Expr lhs, out Expr rhs);
  }

  class IntegerRepresentationHelper {
    internal static bool IsFun(Expr e, string mneumonic, out Expr lhs, out Expr rhs)
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

      if (!Regex.IsMatch(fc.FunctionName, "BV[0-9]+_" + mneumonic))
      {
        return false;
      }

      lhs = ne.Args[0];
      rhs = ne.Args[1];
      return true;
    }
  }

  class BVIntegerRepresentation : IntegerRepresentation {

    private GPUVerifier verifier;

    public BVIntegerRepresentation(GPUVerifier verifier) {
      this.verifier = verifier;
    }

    public Microsoft.Boogie.Type GetIntType(int Width) {
      return Microsoft.Boogie.Type.GetBvType(Width);
    }

    public LiteralExpr GetLiteral(int Value, int Width) {
      return new LiteralExpr(Token.NoToken, BigNum.FromInt(Value), Width);
    }

    private Expr MakeBitVectorBinaryBoolean(string suffix, string smtName, Expr lhs, Expr rhs)
    {
        return MakeBVFunctionCall("BV" + lhs.Type.BvBits + "_" + suffix, smtName, Microsoft.Boogie.Type.Bool, lhs, rhs);
    }

    private Expr MakeBitVectorUnaryBitVector(string suffix, string smtName, Expr expr, Microsoft.Boogie.Type resultType)
    {
        return MakeBVFunctionCall("BV" + expr.Type.BvBits + "_" + suffix, smtName, resultType, expr);
    }

    private Expr MakeBitVectorBinaryBitVector(string suffix, string smtName, Expr lhs, Expr rhs)
    {
        return MakeBVFunctionCall("BV" + lhs.Type.BvBits + "_" + suffix, smtName, lhs.Type, lhs, rhs);
    }

    private Expr MakeBVFunctionCall(string functionName, string smtName, Microsoft.Boogie.Type resultType, params Expr[] args)
    {
        Function f = verifier.GetOrCreateBVFunction(functionName, smtName, resultType, args.Select(a => a.Type).ToArray());
        var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr>(args));
        e.Type = resultType;
        return e;
    }

    public Expr MakeSub(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBitVector("SUB", "bvsub", lhs, rhs);
    }

    public Expr MakeAnd(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBitVector("AND", "bvand", lhs, rhs);
    }

    public Expr MakeSge(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBoolean("SGE", "bvsge", lhs, rhs);
    }

    public Expr MakeSlt(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBoolean("SLT", "bvslt", lhs, rhs);
    }

    public Expr MakeSle(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBoolean("SLE", "bvsle", lhs, rhs);
    }

    public Expr MakeSgt(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBoolean("SGT", "bvsgt", lhs, rhs);
    }

    public Expr MakeAdd(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBitVector("ADD", "bvadd", lhs, rhs);
    }

    public Expr MakeMul(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBitVector("MUL", "bvmul", lhs, rhs);
    }

    public Expr MakeDiv(Expr lhs, Expr rhs) {
      return MakeBitVectorBinaryBitVector("DIV", "bvsdiv", lhs, rhs);
    }

    public Expr MakeModPow2(Expr lhs, Expr rhs) {
      var BVType = rhs.Type as BvType;
      return MakeAnd(MakeSub(rhs, GetLiteral(1, BVType.Bits)), lhs);
    }

    public Expr MakeZext(Expr expr, Microsoft.Boogie.Type resultType)
    {
      if (expr.Type.BvBits == resultType.BvBits)
        return expr;
      else
        return MakeBitVectorUnaryBitVector("ZEXT" + resultType.BvBits, "zero_extend " + (resultType.BvBits - expr.Type.BvBits), expr, resultType);
    }

    public bool IsAdd(Expr e, out Expr lhs, out Expr rhs) {
      return IntegerRepresentationHelper.IsFun(e, "ADD", out lhs, out rhs);
    }

    public bool IsMul(Expr e, out Expr lhs, out Expr rhs) {
      return IntegerRepresentationHelper.IsFun(e, "MUL", out lhs, out rhs);
    }

  }

  class MathIntegerRepresentation : IntegerRepresentation {

    private GPUVerifier verifier;

    public MathIntegerRepresentation(GPUVerifier verifier) {
      this.verifier = verifier;
    }

    public Microsoft.Boogie.Type GetIntType(int Width) {
      return Microsoft.Boogie.Type.Int;
    }

    public LiteralExpr GetLiteral(int Value, int Width) {
      return new LiteralExpr(Token.NoToken, BigNum.FromInt(Value));
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
        return MakeIntFunctionCall("BV32_" + suffix, infixOp, Microsoft.Boogie.Type.Bool, lhs, rhs);
    }

    private Expr MakeIntFunctionCall(string functionName, BinaryOperator.Opcode infixOp, Microsoft.Boogie.Type resultType, Expr lhs, Expr rhs)
    {
        Function f = verifier.GetOrCreateIntFunction(functionName, infixOp, resultType, lhs.Type, rhs.Type);
        var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr> { lhs, rhs });
        e.Type = resultType;
        return e;
    }

    private Expr MakeIntUFFunctionCall(string functionName, Microsoft.Boogie.Type resultType, Expr lhs, Expr rhs)
    {
        Function f = verifier.GetOrCreateBinaryUF(functionName, resultType, lhs.Type, rhs.Type);
        var e = new NAryExpr(Token.NoToken, new FunctionCall(f), new List<Expr> { lhs, rhs });
        e.Type = resultType;
        return e;
    }


    public Expr MakeSub(Expr lhs, Expr rhs) {
      return MakeIntBinaryInt("SUB", BinaryOperator.Opcode.Sub, lhs, rhs);
    }

    public Expr MakeAdd(Expr lhs, Expr rhs) {
      return MakeIntBinaryInt("ADD", BinaryOperator.Opcode.Add, lhs, rhs);
    }

    public Expr MakeMul(Expr lhs, Expr rhs) {
      return MakeIntBinaryInt("MUL", BinaryOperator.Opcode.Mul, lhs, rhs);
    }

    public Expr MakeDiv(Expr lhs, Expr rhs) {
      return MakeIntBinaryInt("MUL", BinaryOperator.Opcode.Div, lhs, rhs);
    }

    public Expr MakeAnd(Expr lhs, Expr rhs) {
      return MakeIntBinaryIntUF("AND", lhs, rhs);
    }

    public Expr MakeSgt(Expr lhs, Expr rhs) {
      return MakeIntBinaryBool("SGT", BinaryOperator.Opcode.Gt, lhs, rhs);
    }

    public Expr MakeSge(Expr lhs, Expr rhs) {
      return MakeIntBinaryBool("SGE", BinaryOperator.Opcode.Ge, lhs, rhs);
    }

    public Expr MakeSlt(Expr lhs, Expr rhs) {
      return MakeIntBinaryBool("SLT", BinaryOperator.Opcode.Lt, lhs, rhs);
    }

    public Expr MakeSle(Expr lhs, Expr rhs) {
      return MakeIntBinaryBool("SLE", BinaryOperator.Opcode.Le, lhs, rhs);
    }

    public Expr MakeModPow2(Expr lhs, Expr rhs) {
      return Expr.Binary(BinaryOperator.Opcode.Mod, lhs, rhs);
    }

    public Expr MakeZext(Expr expr, Microsoft.Boogie.Type resultType)
    {
      return expr;
    }

    public bool IsAdd(Expr e, out Expr lhs, out Expr rhs) {
      return IntegerRepresentationHelper.IsFun(e, "ADD", out lhs, out rhs);
    }

    public bool IsMul(Expr e, out Expr lhs, out Expr rhs) {
      return IntegerRepresentationHelper.IsFun(e, "MUL", out lhs, out rhs);
    }

  }

}
