using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
		private static GPU gpu = new GPU();
		private static Memory Memory = new Memory();
		private static Dictionary<Expr, ExprTree> ExprTrees = new Dictionary<Expr, ExprTree>(); 
		private static Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
		private static HashSet<AssertCmd> failedAsserts = new HashSet<AssertCmd>();
		private static BitVector32 False = new BitVector32(0);
		private static BitVector32 True = new BitVector32(1);
		
		public static void Interpret (Program program)
		{
			IEnumerable<Axiom> axioms = program.TopLevelDeclarations.OfType<Axiom>();
			foreach (Axiom axiom in axioms)
			{
				ExprTree tree = new ExprTree(axiom.Expr);
			}
			
			IEnumerable<Constant> constants = program.TopLevelDeclarations.OfType<Constant>();
			foreach (Constant constant in constants)
			{
				bool existential = false;
				if (constant.CheckBooleanAttribute("existential", ref existential))
				{
					if (existential)
						Memory.Store(constant.Name, True);
					else
						Memory.Store(constant.Name, False);
				}
				else if (GPU.IsLocalIDName(constant.Name))
					Memory.Store(constant.Name, new BitVector32(Random.Next(1, 10)));
			}
			
			IEnumerable<Implementation> implementations = program.TopLevelDeclarations.OfType<Implementation>().
				Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "kernel"));
			foreach (Implementation impl in implementations)
			{
				Print.VerboseMessage(String.Format("Interpreting implementation '{0}'", impl.Name));
				LabelToBlock.Clear();
				foreach (Block block in impl.Blocks)
				{
					LabelToBlock[block.Label] = block;
				}
				InitialiseFormalParams(impl.InParams);
				Block entry = impl.Blocks[0];
				InterpretBasicBlock(entry);
				Memory.Dump();
			}
			
			if (failedAsserts.Count > 0)
			{
				Console.WriteLine("The following asserts were disproven");
				foreach (AssertCmd assert in failedAsserts)
				{
					Console.WriteLine(assert.ToString());
				}
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
		
		private static void EvaluateExprTree (ExprTree tree)
		{			
			foreach (HashSet<Node> nodes in tree)
			{
				foreach (Node node in nodes)
				{
					if (node is ScalarSymbolNode<BitVector32>)
					{
						ScalarSymbolNode<BitVector32> scalar = node as ScalarSymbolNode<BitVector32>;
						scalar.evaluation = Memory.GetValue(scalar.symbol);
					}
					else if (node is ScalarSymbolNode<bool>)
					{
						ScalarSymbolNode<bool> scalar = node as ScalarSymbolNode<bool>;
						if (Memory.Contains(scalar.symbol) ||
						    !(tree.Root() is TernaryNode<bool> || tree.Root() is TernaryNode<BitVector32>))
						{
							if (Memory.GetValue(scalar.symbol).Equals(True))
								scalar.evaluation = true;
							else
								scalar.evaluation = false;
						}
					}
					else if (node is MapSymbolNode<BitVector32>)
					{
						MapSymbolNode<BitVector32> map = node as MapSymbolNode<BitVector32>;
						SubscriptExpr subscriptExpr = new SubscriptExpr();
						foreach (ExprNode<BitVector32> child in map.GetChildren())
						{
							BitVector32 subscript = child.evaluation;
							subscriptExpr.AddIndex(subscript);
						}
						map.evaluation = Memory.GetValue(map.basename, subscriptExpr);
					}
					else if (node is MapSymbolNode<bool>)
					{
						MapSymbolNode<bool> map = node as MapSymbolNode<bool>;
						Print.ExitMessage("Map: " + map.ToString());
					}
					else if (node is UnaryNode<bool>)
					{
						UnaryNode<bool> unary = node as UnaryNode<bool>;
						ExprNode<bool> child  = (ExprNode<bool>) unary.GetChildren()[0];
						switch (unary.op)
						{
						case "!":
						{
							unary.evaluation = !child.evaluation;
							break;
						}
						}
					}
					else if (node is BinaryNode<BitVector32>)
					{
						BinaryNode<BitVector32> binary = node as BinaryNode<BitVector32>;
						ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
						ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
						switch (binary.op)
						{
						case "+":
							binary.evaluation = new BitVector32(left.evaluation.Data + right.evaluation.Data);
							break;
						case "-":
							binary.evaluation = new BitVector32(left.evaluation.Data - right.evaluation.Data);
							break;
						case "*":
							binary.evaluation = new BitVector32(left.evaluation.Data * right.evaluation.Data);
							break;
						case "/":
							binary.evaluation = new BitVector32(left.evaluation.Data / right.evaluation.Data);
							break;
						default:
							Print.ExitMessage("Unhandled bv binary op: " + binary.op);
							break;
						}
					}
					else if (node is BinaryNode<bool>)
					{
						BinaryNode<bool> binary = node as BinaryNode<bool>;
						switch (binary.op)
						{
						case "<":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data < right.evaluation.Data;
							break;
						}
						case "<=":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data <= right.evaluation.Data;
							break;
						}
						case ">":							
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data > right.evaluation.Data;
							break;
						}
						case ">=":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data >= right.evaluation.Data;
							break;
						}
						case "==":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data == right.evaluation.Data;
							break;
						}
						case "!=":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) binary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation.Data != right.evaluation.Data;
							break;
						}
						case "||":
						{
							ExprNode<bool> left  = (ExprNode<bool>) binary.GetChildren()[0];
							ExprNode<bool> right = (ExprNode<bool>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation || right.evaluation;
							break;
						}
						case "&&":
						{
							ExprNode<bool> left  = (ExprNode<bool>) binary.GetChildren()[0];
							ExprNode<bool> right = (ExprNode<bool>) binary.GetChildren()[1];
							binary.evaluation = left.evaluation && right.evaluation;
							break;
						}
						case "==>":
						{
							ExprNode<bool> left  = (ExprNode<bool>) binary.GetChildren()[0];
							ExprNode<bool> right = (ExprNode<bool>) binary.GetChildren()[1];
							if (left.evaluation && !right.evaluation)
								binary.evaluation = false;
							else
								binary.evaluation = true;
							break;
						}
						default:
						{
							Print.ExitMessage("Unhandled bool binary op: " + binary.op);
							break;
						}
						}
					}
					else if (node is TernaryNode<bool>)
					{
						TernaryNode<bool> ternary = node as TernaryNode<bool>;
						ExprNode<bool> one   = (ExprNode<bool>) ternary.GetChildren()[0];
						ExprNode<bool> two   = (ExprNode<bool>) ternary.GetChildren()[1];
						ExprNode<bool> three = (ExprNode<bool>) ternary.GetChildren()[2];
						if (one.evaluation)
							ternary.evaluation = two.evaluation;
						else
							ternary.evaluation = three.evaluation;
					}
					else if (node is TernaryNode<BitVector32>)
					{
						TernaryNode<BitVector32> ternary = node as TernaryNode<BitVector32>;
						ExprNode<bool> one          = (ExprNode<bool>) ternary.GetChildren()[0];
						ExprNode<BitVector32> two   = (ExprNode<BitVector32>) ternary.GetChildren()[1];
						ExprNode<BitVector32> three = (ExprNode<BitVector32>) ternary.GetChildren()[2];
						if (one.evaluation)
							ternary.evaluation = two.evaluation;
						else
							ternary.evaluation = three.evaluation;
					}
					else if (node is NaryNode<BitVector32>)
					{
						NaryNode<BitVector32> nary = node as NaryNode<BitVector32>;
						switch (nary.op)
						{
						case "BV32_SUB":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = new BitVector32(left.evaluation.Data - right.evaluation.Data);
							break;
						}
						case "BV32_ADD":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = new BitVector32(left.evaluation.Data + right.evaluation.Data);
							break;
						}
						case "BV32_AND":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = new BitVector32(left.evaluation.Data & right.evaluation.Data);
							break;
						}
						default:
						{
							Print.ExitMessage("Unhandled bv nary op: " + nary.op);
							break;
						}
						}
					}
					else if (node is NaryNode<bool>)
					{
						NaryNode<bool> nary = node as NaryNode<bool>;
						switch (nary.op)
						{
						case "BV32_SLT":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = left.evaluation.Data < right.evaluation.Data;
							break;
						}
						case "BV32_SLE":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = left.evaluation.Data <= right.evaluation.Data;
							break;
						}
						case "BV32_SGT":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = left.evaluation.Data > right.evaluation.Data;
							break;
						}
						case "BV32_SGE":
						{
							ExprNode<BitVector32> left  = (ExprNode<BitVector32>) nary.GetChildren()[0];
							ExprNode<BitVector32> right = (ExprNode<BitVector32>) nary.GetChildren()[1];
							nary.evaluation = left.evaluation.Data >= right.evaluation.Data;
							break;
						}
						default:
						{
							Print.ExitMessage("Unhandled bool nary op: " + nary.op);
							break;
						}
						}
					}
				}
			}
			
			Node root = tree.Root();
			if (root is ExprNode<bool>)
			{
				ExprNode<bool> boolRoot = root as ExprNode<bool>;
				if (boolRoot.evaluation)
					tree.evaluation = True;
				else
					tree.evaluation = False;
			}
			else
			{
				ExprNode<BitVector32> bvRoot = root as ExprNode<BitVector32>;
				tree.evaluation = bvRoot.evaluation;
			}
		}
		
		private static ExprTree GetExprTree (Expr expr)
		{
			if (!ExprTrees.ContainsKey(expr))
				ExprTrees[expr] = new ExprTree(expr);
			return ExprTrees[expr];
		}

		private static void InterpretBasicBlock (Block block)
		{
			Print.VerboseMessage(String.Format("Entering basic block with label '{0}'", block.Label));
			// Execute all the statements
			foreach (Cmd cmd in block.Cmds)
			{
				if (cmd is AssignCmd)
				{
					AssignCmd assign = cmd as AssignCmd;
					List<ExprTree> evaluations = new List<ExprTree>();
					// First evaluate all RHS expressions
					foreach (Expr expr in assign.Rhss) 
					{ 
						ExprTree exprTree = GetExprTree(expr);
						EvaluateExprTree(exprTree);
						evaluations.Add(exprTree);
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
								ExprTree exprTree = GetExprTree(index);
								EvaluateExprTree(exprTree);
								BitVector32 subscript = exprTree.evaluation;
								subscriptExpr.AddIndex(subscript);
							}
							ExprTree tree = LhsEval.Item2;
							Memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree.evaluation);
						}
						else
						{
							SimpleAssignLhs lhs = (SimpleAssignLhs) LhsEval.Item1;
							ExprTree tree       = LhsEval.Item2;
							Memory.Store(lhs.AssignedVariable.Name, tree.evaluation);
						}
					}
					
					Print.VerboseMessage(assign.ToString());
					Memory.Dump();
				}
				if (cmd is AssertCmd)
				{
					AssertCmd assert = cmd as AssertCmd;
					ExprTree exprTree = GetExprTree(assert.Expr);
					EvaluateExprTree(exprTree);
					if (exprTree.evaluation.Equals(False))
					{
						Print.VerboseMessage("Falsifying assertion: " + assert.ToString());
						failedAsserts.Add(assert);
					}
					Memory.Dump();
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
						ExprTree exprTree         = GetExprTree(predicateCmd.Expr);
						EvaluateExprTree(exprTree);
						if (exprTree.evaluation.Equals(True))
						{
							InterpretBasicBlock(succ);
							found = true;
							break;
						}
					}
					if (!found)
						Print.ExitMessage("No successor guard evaluates to true");
				}
			}
			else if (transfer is ReturnCmd)
				Print.VerboseMessage("Execution done");
			else
				Print.ExitMessage("Unhandled control transfer command: " + transfer.ToString());
		}
	}
}

