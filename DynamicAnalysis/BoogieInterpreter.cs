using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
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
					Implementation impl = decl as Implementation;
					Print.VerboseMessage(String.Format("Found implementation '{0}'", impl.Name));
					buildLabelMap(impl);
					allocateFormalParams(impl.InParams);
					Block entry = impl.Blocks[0];
					interpretBasicBlock(entry);
					memory.dump();
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
		
		private static void allocateFormalParams (List<Variable> formals)
		{
			foreach (Variable v in formals)
			{
				Print.VerboseMessage(String.Format("Found formal parameter '{0}'", v.Name));
				memory.store(v.Name, random.Next(1, 100));
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
						memory.store(lhsName, evaluateIntExpr(LhsRhs.Item2));
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
				GotoCmd goto_ = transfer as GotoCmd;
				foreach (string succLabel in goto_.labelNames)
				{
					Block succ = labelToBlock[succLabel];
					PredicateCmd predicateCmd = (PredicateCmd) succ.Cmds[0];
					bool val = evaluateBoolExpr((NAryExpr) predicateCmd.Expr);
					if (val)
						interpretBasicBlock(succ);
				}
			}
			if (transfer is ReturnCmd)
			{
				ReturnCmd return_ = (ReturnCmd) transfer;
				Print.VerboseMessage("Returning");
			}
			if (transfer is ReturnExprCmd)
			{
				ReturnExprCmd return_ = (ReturnExprCmd) transfer;
			}
		}
		
		private static int evaluateIntExpr (Expr expr)
		{
			if (expr is IdentifierExpr)
			{
				IdentifierExpr ident = (IdentifierExpr) expr;
				IntVal val           = memory.getValue(ident.Name);
				return val.getVal();
			}
			if (expr is LiteralExpr)
			{
				LiteralExpr literal = (LiteralExpr) expr;
				if (literal.isBigNum)
				{
					int val = literal.asBigNum.ToIntSafe;
					return val;
				}
			}
			return 0;
		}
		
		private static bool evaluateBoolExpr (NAryExpr nary)
		{
			BinaryOperator binOp = (BinaryOperator) nary.Fun;
			int lhs              = evaluateIntExpr(nary.Args[0]);
			int rhs              = evaluateIntExpr(nary.Args[1]);
			switch (binOp.Op)
			{
			case BinaryOperator.Opcode.Eq:
				return lhs == rhs;
			case BinaryOperator.Opcode.Le:
				return lhs <= rhs;
			case BinaryOperator.Opcode.Lt:
				return lhs < rhs;
			case BinaryOperator.Opcode.Ge:
				return lhs >= rhs;
			case BinaryOperator.Opcode.Gt:
				return lhs > rhs;
			}
			return false;
		}
	}
}

