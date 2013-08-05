using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Collections.Specialized;
using System.Text.RegularExpressions;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace DynamicAnalysis
{
	public class BoogieInterpreter
	{
		private static Random Random = new Random();
		private static Memory Memory = new Memory();
		private static Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
		
		public static void Interpret (Program program)
		{
			foreach (var decl in program.TopLevelDeclarations)
			{
				if (decl is Implementation)
				{
					Memory.Clear();
					LabelToBlock.Clear();
					Implementation impl = decl as Implementation;
					Print.VerboseMessage(String.Format("Found implementation '{0}'", impl.Name));
					BuildLabelMap(impl);
					InitialiseFormalParams(impl.InParams);
					Block entry = impl.Blocks[0];
					InterpretBasicBlock(entry);
					Memory.Dump();
				}
				if (decl is DeclWithFormals)
				{
					DeclWithFormals functionDecl = decl as DeclWithFormals;
					Print.VerboseMessage(String.Format("Found function declaration '{0}'", functionDecl.Name));
				}
				if (decl is Constant)
				{
					Constant const_ = decl as Constant;
					Print.VerboseMessage(const_.Name);
				}
				if (decl is Axiom)
				{
					Axiom axiom = decl as Axiom;
					Print.VerboseMessage(axiom.Expr.ToString());
				}
			}
		}
		
		private static void BuildLabelMap (Implementation impl)
		{
			foreach (Block block in impl.Blocks)
			{
				LabelToBlock[block.Label] = block;
			}
		}
		
		private static void InitialiseFormalParams (List<Variable> formals)
		{
			foreach (Variable v in formals)
			{
				Print.VerboseMessage(String.Format("Found formal parameter '{0}' with type '{1}'", v.Name, v.TypedIdent.Type.ToString()));
				if (v.TypedIdent.Type is BvType)
				{
					BvType bv = (BvType) v.TypedIdent.Type;
					if (bv.Bits == 1)
						Memory.Store(v.Name, new BitVector32(Random.Next(0, 1)));
					else
					{	
						int lowestVal  = (int) -Math.Pow(2, bv.Bits-1);
						int highestVal = (int) Math.Pow(2, bv.Bits-1) - 1;
						Memory.Store(v.Name, new BitVector32(Random.Next(lowestVal, highestVal)));
					}
				}
				else if (v.TypedIdent.Type is BasicType)
				{
					BasicType basic = (BasicType) v.TypedIdent.Type;
					if (basic.IsInt)
					{
						int lowestVal  = (int) -Math.Pow(2, 32-1);
						int highestVal = (int) Math.Pow(2, 32-1) - 1;
						Memory.Store(v.Name, new BitVector32(Random.Next(lowestVal, highestVal)));
					}
					else
						Print.ExitMessage(String.Format("Unhandled basic type '{0}'", basic.ToString()));
				}
				else
					Print.ExitMessage("Unknown data type");
			}
		}

		private static void InterpretBasicBlock (Block block)
		{
			Print.VerboseMessage(String.Format("Entering basic block with label '{0}'", block.Label));
			// Execute all the statements
			foreach (Cmd cmd in block.Cmds)
			{
				if (cmd is AssignCmd)
				{
					AssignCmd assign = (AssignCmd) cmd;
					List<BitVector32> evaluations = new List<BitVector32>();
					// First evaluate all RHS expressions
					foreach (Expr expr in assign.Rhss) 
					{ 
						evaluations.Add(EvaluateArithmeticExpr(expr));
					}
					// Now update the store
					foreach (var LhsEval in assign.Lhss.Zip(evaluations)) 
					{
						if (LhsEval.Item1 is MapAssignLhs)
						{
							MapAssignLhs lhs = (MapAssignLhs) LhsEval.Item1;
							SubscriptExpr subscriptExpr = new SubscriptExpr();
							foreach (Expr index in lhs.Indexes)
							{
								BitVector32 subscript = EvaluateArithmeticExpr(index);
								subscriptExpr.AddIndex(subscript);
							}
							Memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, LhsEval.Item2);
						}
						else
						{
							SimpleAssignLhs lhs = (SimpleAssignLhs) LhsEval.Item1;
							Memory.Store(lhs.AssignedVariable.Name, LhsEval.Item2);
						}
					}
				}
			}
			// Now transfer control
			TransferControl(block);
		}
		
		private static void TransferControl (Block block)
		{
			TransferCmd transfer = block.TransferCmd;
			if (transfer is GotoCmd)
			{
				bool found    = false;
				GotoCmd goto_ = transfer as GotoCmd;
				if (goto_.labelNames.Count == 1)
				{
					string succLabel = goto_.labelNames[0];
					Block succ       = LabelToBlock[succLabel];
					InterpretBasicBlock(succ);
				}
				else
				{
					// Loop through all potential successors and find one whose guard evaluates to true
					foreach (string succLabel in goto_.labelNames)
					{
						Block succ                = LabelToBlock[succLabel];
						PredicateCmd predicateCmd = (PredicateCmd) succ.Cmds[0];
						found                     = EvaluateBoolExpr(predicateCmd.Expr);
						if (found)
						{
							InterpretBasicBlock(succ);
							break;
						}
					}
					if (!found)
						Print.ExitMessage("No successor guard evaluates to true");
				}
			}
			if (transfer is ReturnCmd)
			{
				ReturnCmd return_ = (ReturnCmd) transfer;
				Print.VerboseMessage("Execution done");
			}
			if (transfer is ReturnExprCmd)
			{
				ReturnExprCmd return_ = (ReturnExprCmd) transfer;
			}
		}
		
		private static BitVector32 EvaluateSymbol (Expr expr)
		{
			if (expr is IdentifierExpr)
			{
				IdentifierExpr ident = (IdentifierExpr) expr;
				return Memory.GetValue(ident.Name);
			}
			if (expr is LiteralExpr)
			{
				LiteralExpr literal = (LiteralExpr) expr;
				if (literal.Val is BvConst)
				{
					BvConst bv = (BvConst) literal.Val;
					return new BitVector32(bv.Value.ToInt);
				}
				else if (literal.Val is BigNum)
				{
					BigNum num = (BigNum) literal.Val;
					return new BitVector32(num.ToInt);
				}
			}
			Print.ExitMessage("Unknown symbol: " + expr.ToString());
			return new BitVector32(0);
		}
		
		private static BitVector32 EvaluateArithmeticExpr (Expr expr)
		{
			if (expr is NAryExpr)
			{
				NAryExpr nary = (NAryExpr) expr;
				if (nary.Fun is BinaryOperator)
				{
					BinaryOperator binary = (BinaryOperator) nary.Fun;
					BitVector32 lhs       = EvaluateArithmeticExpr(nary.Args[0]);
					BitVector32 rhs       = EvaluateArithmeticExpr(nary.Args[1]);
					switch (binary.Op)
					{
					case BinaryOperator.Opcode.Add:
						return new BitVector32(lhs.Data + rhs.Data);
					case BinaryOperator.Opcode.Mul:
						return new BitVector32(lhs.Data * rhs.Data);
					case BinaryOperator.Opcode.Sub:
						return new BitVector32(lhs.Data - rhs.Data);
					}
				}
				else if (nary.Fun is MapSelect)
				{
					IdentifierExpr basename = (IdentifierExpr) nary.Args[0];
					SubscriptExpr subscriptExpr = new SubscriptExpr();
					foreach (Expr index in nary.Args.GetRange(1, nary.Args.Count - 1))
					{
						BitVector32 subscript = EvaluateArithmeticExpr(index);
						subscriptExpr.AddIndex(subscript);
					}
					return Memory.GetValue(basename.Name, subscriptExpr);
				}
			}
			return EvaluateSymbol(expr);
		}
		
		private static bool IsLogicalBinaryOp (BinaryOperator.Opcode op)
		{
			return op == BinaryOperator.Opcode.Or || op == BinaryOperator.Opcode.And;
		}
		
		private static bool EvaluateBoolExpr (Expr expr)
		{
			if (expr is LiteralExpr)
			{
				LiteralExpr literal = (LiteralExpr) expr;
				return literal.IsTrue;
			}
			else if (expr is NAryExpr)
			{
				NAryExpr nary = (NAryExpr) expr;				
				if (nary.Fun is BinaryOperator)
				{
					BinaryOperator binary = (BinaryOperator) nary.Fun;
					if (IsLogicalBinaryOp(binary.Op))
					{
						bool lhs = EvaluateBoolExpr(nary.Args[0]);
						bool rhs = EvaluateBoolExpr(nary.Args[1]);
						if (binary.Op == BinaryOperator.Opcode.Or)
							return lhs || rhs;
						else if (binary.Op == BinaryOperator.Opcode.And)
							return lhs && rhs;
					}
					else
					{
						BitVector32 lhs = EvaluateSymbol(nary.Args[0]);
						BitVector32 rhs = EvaluateSymbol(nary.Args[1]);
						switch (binary.Op)
						{
						case BinaryOperator.Opcode.Eq:
							return lhs.Data == rhs.Data;
						case BinaryOperator.Opcode.Le:
							return lhs.Data <= rhs.Data;
						case BinaryOperator.Opcode.Lt:
							return lhs.Data < rhs.Data;
						case BinaryOperator.Opcode.Ge:
							return lhs.Data >= rhs.Data;
						case BinaryOperator.Opcode.Gt:
							return lhs.Data > rhs.Data;
						default:
							Print.ExitMessage("Unhandled binary operator");
							return false;
						}
					}
				}
				else if (nary.Fun is UnaryOperator)
				{
					UnaryOperator unary = (UnaryOperator) nary.Fun;
					bool arg            = EvaluateBoolExpr(nary.Args[0]);
					switch (unary.Op)
					{
					case UnaryOperator.Opcode.Not:
						return !arg;
					default:
						Print.ExitMessage("Unhandled unary operator");
						return false;
					}
				}
				else if (nary.Fun is FunctionCall)
				{
					FunctionCall call = (FunctionCall) nary.Fun;
					if (Regex.IsMatch(call.FunctionName, "BV32", RegexOptions.IgnoreCase))
					{
						BitVector32 lhs = EvaluateSymbol(nary.Args[0]);
						BitVector32 rhs = EvaluateSymbol(nary.Args[1]);
						switch (call.FunctionName)
						{
						case "BV32_GT":
							return lhs.Data > rhs.Data;
						case "BV32_LT":
							return lhs.Data < rhs.Data;
						default:
							Print.ExitMessage("Unhandled BV operator");
							return false;
						}
					}
					else
						Print.ExitMessage("Unknown BV function call");
				}
			}
			Print.ExitMessage("Unkknow Boolean expression: " + expr.ToString());
			return false;
		}
	}
}

