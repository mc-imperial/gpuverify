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
		private static Random random = new Random();
		private static Memory memory = new Memory();
		private static Dictionary<string, Block> labelToBlock = new Dictionary<string, Block>();
		
		public static void interpret (Program program)
		{
			foreach (var decl in program.TopLevelDeclarations)
			{
				if (decl is Implementation)
				{
					memory.clear();
					labelToBlock.clear();
					Implementation impl = decl as Implementation;
					Print.VerboseMessage(String.Format("Found implementation '{0}'", impl.Name));
					buildLabelMap(impl);
					initialiseFormalParams(impl.InParams);
					Block entry = impl.Blocks[0];
					interpretBasicBlock(entry);
					memory.dump();
				}
				if (decl is DeclWithFormals)
				{
					DeclWithFormals functionDecl = decl as DeclWithFormals;
					Print.VerboseMessage(String.Format("Found function declaration '{0}'", functionDecl.Name));
				}
			}
		}
		
		private static void buildLabelMap (Implementation impl)
		{
			foreach (Block block in impl.Blocks)
			{
				labelToBlock[block.Label] = block;
			}
		}
		
		private static void initialiseFormalParams (List<Variable> formals)
		{
			foreach (Variable v in formals)
			{
				Print.VerboseMessage(String.Format("Found formal parameter '{0}' with type '{1}'", v.Name, v.TypedIdent.Type.ToString()));
				if (v.TypedIdent.Type is BvType)
				{
					BvType bv = (BvType) v.TypedIdent.Type;
					if (bv.Bits == 1)
						memory.store(v.Name, new BitVector32(random.Next(0, 1)));
					else
					{	
						int lowestVal  = (int) -Math.Pow(2, bv.Bits-1);
						int highestVal = (int) Math.Pow(2, bv.Bits-1) - 1;
						memory.store(v.Name, new BitVector32(random.Next(lowestVal, highestVal)));
					}
				}
				else if (v.TypedIdent.Type is BasicType)
				{
					BasicType basic = (BasicType) v.TypedIdent.Type;
					if (basic.IsInt)
					{
						int lowestVal  = (int) -Math.Pow(2, 32-1);
						int highestVal = (int) Math.Pow(2, 32-1) - 1;
						memory.store(v.Name, new BitVector32(random.Next(lowestVal, highestVal)));
					}
					else
						Print.ExitMessage(String.Format("Unhandled basic type '{0}'", basic.ToString()));
				}
				else
					Print.ExitMessage("Unknown data type");
			}
		}

		private static void interpretBasicBlock (Block block)
		{
			Print.VerboseMessage(String.Format("Entering basic block with label '{0}'", block.Label));
			// Execute all the statements
			foreach (Cmd cmd in block.Cmds)
			{
				if (cmd is AssignCmd)
				{
					AssignCmd assign = (AssignCmd) cmd;
					foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) 
					{
			            SimpleAssignLhs lhs = (SimpleAssignLhs) LhsRhs.Item1;
						string lhsName      = lhs.AssignedVariable.Name;
						memory.store(lhsName, evaluateArithmeticExpr(LhsRhs.Item2));
					}
				}
			}
			// Now transfer control
			transferControl(block);
		}
		
		private static void transferControl (Block block)
		{
			TransferCmd transfer = block.TransferCmd;
			if (transfer is GotoCmd)
			{
				bool found    = false;
				GotoCmd goto_ = transfer as GotoCmd;
				if (goto_.labelNames.Count == 1)
				{
					string succLabel = goto_.labelNames[0];
					Block succ       = labelToBlock[succLabel];
					interpretBasicBlock(succ);
				}
				else
				{
					// Loop through all potential successors and find one whose guard evaluates to true
					foreach (string succLabel in goto_.labelNames)
					{
						Block succ                = labelToBlock[succLabel];
						PredicateCmd predicateCmd = (PredicateCmd) succ.Cmds[0];
						found                     = evaluateBoolExpr(predicateCmd.Expr);
						if (found)
						{
							interpretBasicBlock(succ);
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
		
		private static BitVector32 evaluateArithmeticExpr (Expr expr)
		{
			if (expr is IdentifierExpr)
			{
				IdentifierExpr ident = (IdentifierExpr) expr;
				return memory.getValue(ident.Name);
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
			if (expr is NAryExpr)
			{
				NAryExpr nary         = (NAryExpr) expr;
				BinaryOperator binary = (BinaryOperator) nary.Fun;
				BitVector32 lhs       = evaluateArithmeticExpr(nary.Args[0]);
				BitVector32 rhs       = evaluateArithmeticExpr(nary.Args[1]);
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
			Print.ExitMessage("Unhandled expression");
			return new BitVector32(0);
		}
		
		private static bool evaluateBoolExpr (Expr expr)
		{
			if (expr is LiteralExpr)
			{
				LiteralExpr literal = (LiteralExpr) expr;
				return literal.IsTrue;
			}
			if (expr is NAryExpr)
				return evaluateNAryBoolExpr((NAryExpr) expr);
			Print.ExitMessage("Unhandled boolean expression");
			return false;
		}
		
		private static bool evaluateNAryBoolExpr (NAryExpr nary)
		{
			if (nary.Fun is BinaryOperator)
			{
				BinaryOperator binary = (BinaryOperator) nary.Fun;
				BitVector32 lhs       = evaluateArithmeticExpr(nary.Args[0]);
				BitVector32 rhs       = evaluateArithmeticExpr(nary.Args[1]);
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
			else if (nary.Fun is UnaryOperator)
			{
				UnaryOperator unary = (UnaryOperator) nary.Fun;
				bool arg            = evaluateBoolExpr(nary.Args[0]);
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
					string op       = call.FunctionName.Substring(call.FunctionName.IndexOf('_')+1);
					BitVector32 lhs = evaluateArithmeticExpr(nary.Args[0]);
					BitVector32 rhs = evaluateArithmeticExpr(nary.Args[1]);
					switch (op)
					{
					case "GT":
						return lhs.Data > rhs.Data;
					case "LT":
						return lhs.Data < rhs.Data;
					default:
						Print.ExitMessage("Unhandled BV operator");
						return false;
					}
				}
				else
				{
					Print.ExitMessage("Unknown function call");
				}
			}
			return false;
		}
	}
}

