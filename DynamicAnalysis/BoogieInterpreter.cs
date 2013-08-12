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
		private static Memory Memory = new Memory();
		private static Dictionary<Expr, ExprTree> ExprTrees = new Dictionary<Expr, ExprTree>(); 
		private static Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
		private static HashSet<AssertCmd> failedAsserts = new HashSet<AssertCmd>();
		private static HashSet<AssertCmd> passedAsserts = new HashSet<AssertCmd>();
		private static BitVector32 False = new BitVector32(0);
		private static BitVector32 True = new BitVector32(1);
		
		public static void Interpret (Program program)
		{
			IEnumerable<Axiom> axioms = program.TopLevelDeclarations.OfType<Axiom>();
			foreach (Axiom axiom in axioms)
			{
				EvaulateAxiom(axiom);
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
				else if (Regex.IsMatch(constant.Name, "local_id_x", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetThreadID(DIMENSION.X))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.threadID[DIMENSION.X]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.blockDim[DIMENSION.X])));
				}
				else if (Regex.IsMatch(constant.Name, "local_id_y", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetThreadID(DIMENSION.Y))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.threadID[DIMENSION.Y]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.blockDim[DIMENSION.Y])));
				}
				else if (Regex.IsMatch(constant.Name, "local_id_z", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetThreadID(DIMENSION.Z))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.threadID[DIMENSION.Z]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.blockDim[DIMENSION.Z])));
				}
				else if (Regex.IsMatch(constant.Name, "group_id_x", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetGroupID(DIMENSION.X))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.groupID[DIMENSION.X]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.gridDim[DIMENSION.X])));
				}
				else if (Regex.IsMatch(constant.Name, "group_id_y", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetGroupID(DIMENSION.Y))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.groupID[DIMENSION.Y]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.gridDim[DIMENSION.Y])));
				}
				else if (Regex.IsMatch(constant.Name, "group_id_z", RegexOptions.IgnoreCase))
				{
					if (GPU.Instance.IsUserSetGroupID(DIMENSION.Z))
						Memory.Store(constant.Name, new BitVector32(GPU.Instance.groupID[DIMENSION.Z]));
					else
						Memory.Store(constant.Name, new BitVector32(Random.Next(1, GPU.Instance.gridDim[DIMENSION.Z])));
				}
			}
			
			Print.VerboseMessage(GPU.Instance.ToString());
			
			IEnumerable<NamedDeclaration> raceVariables = program.TopLevelDeclarations.OfType<NamedDeclaration>().
				Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "race_checking"));
			foreach (NamedDeclaration decl in raceVariables)
			{
				if (IsRaceArrayOffsetVariable(decl.Name))
					Memory.AddRaceArrayVariable(decl.Name);
			}
			
			IEnumerable<Implementation> implementations = program.TopLevelDeclarations.OfType<Implementation>().
				Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "kernel"));
			foreach (Implementation impl in implementations)
			{
				Print.VerboseMessage(String.Format("Interpreting implementation '{0}'", impl.Name));
				foreach (Requires requires in impl.Proc.Requires)
				{
					EvaluateRequires(requires, raceVariables);
				}
				Print.DebugMessage(Memory.Dump, 5);
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
				Console.WriteLine("The following asserts do NOT hold");
				foreach (AssertCmd assert in failedAsserts)
					Console.WriteLine(assert.ToString());
			}
			if (passedAsserts.Count > 0)
			{
				Console.WriteLine("The following asserts HOLD");
				foreach (AssertCmd assert in passedAsserts)
					Console.WriteLine(assert.ToString());
			}
		}
		
		private static bool IsRaceArrayOffsetVariable (string name)
		{
			return Regex.IsMatch(name, "_(WRITE|READ|ATOMIC)_OFFSET_", RegexOptions.IgnoreCase);
		}
				
		private static ExprTree GetExprTree (Expr expr)
		{
			if (!ExprTrees.ContainsKey(expr))
				ExprTrees[expr] = new ExprTree(expr);
			ExprTrees[expr].ClearEvaluations();
			return ExprTrees[expr];
		}
		
		private static void EvaulateAxiom (Axiom axiom)
		{
			ExprTree tree = GetExprTree(axiom.Expr);
			Stack<Node> stack = new Stack<Node>();
			stack.Push(tree.Root());
			bool search = true;
			while (search && stack.Count > 0)
			{
				Node node = stack.Pop();
				if (node is BinaryNode<bool>)
				{
					BinaryNode<bool> binary = (BinaryNode<bool>) node;
					if (binary.op == "==")
					{
						// Assume that equality is actually assignment into the variable of interest
						search = false;
						ScalarSymbolNode<BitVector32> left = (ScalarSymbolNode<BitVector32>) binary.GetChildren()[0];
						LiteralNode<BitVector32> right     = (LiteralNode<BitVector32>) binary.GetChildren()[1];
						if (left.symbol == "group_size_x") 
						{
							if (!GPU.Instance.IsUserSetBlockDim(DIMENSION.X))
								GPU.Instance.blockDim[DIMENSION.X] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.blockDim[DIMENSION.X]));
						}
						else if (left.symbol == "group_size_y")
						{
							if (!GPU.Instance.IsUserSetBlockDim(DIMENSION.Y))
								GPU.Instance.blockDim[DIMENSION.Y] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.blockDim[DIMENSION.Y]));
						}
						else if (left.symbol == "group_size_z")
						{
							if (!GPU.Instance.IsUserSetBlockDim(DIMENSION.Z))
								GPU.Instance.blockDim[DIMENSION.Z] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.blockDim[DIMENSION.Z]));
						}
						else if (left.symbol == "num_groups_x")
						{
							if (!GPU.Instance.IsUserSetGridDim(DIMENSION.X))
								GPU.Instance.gridDim[DIMENSION.X] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.gridDim[DIMENSION.X]));
						}
						else if (left.symbol == "num_groups_y")
						{
							if (!GPU.Instance.IsUserSetGridDim(DIMENSION.Y))
								GPU.Instance.gridDim[DIMENSION.Y] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.gridDim[DIMENSION.Y]));
						}
						else if (left.symbol == "num_groups_z")
						{
							if (!GPU.Instance.IsUserSetGridDim(DIMENSION.Z))
								GPU.Instance.gridDim[DIMENSION.Z] = right.evaluations[0].Data;
							Memory.Store(left.symbol, new BitVector32(GPU.Instance.gridDim[DIMENSION.Z]));
						}
						else
							Print.ExitMessage("Unhandled GPU axiom: " + axiom.ToString());
					}
				}
				foreach (Node child in node.GetChildren())
					stack.Push(child);
			}
		}
		
		private static void EvaluateRequires (Requires requires, IEnumerable<NamedDeclaration> raceVariables)
		{
			ExprTree tree = new ExprTree(requires.Condition);
			foreach (HashSet<Node> nodes in tree)
			{	
				foreach (Node node in nodes)
				{
					if (node is ScalarSymbolNode<bool>)
					{
						// Initially assume the boolean variable holds. If it is negated this will be handled
						// further up in the expression tree
						ScalarSymbolNode<bool> scalar = (ScalarSymbolNode<bool>) node;
						Memory.Store(scalar.symbol, True);
					}
					else if (node is UnaryNode<bool>)
					{
						UnaryNode<bool> unary = node as UnaryNode<bool>;
						ScalarSymbolNode<bool> child = (ScalarSymbolNode<bool>) unary.GetChildren()[0];
						switch (unary.op)
						{
						case "!":
						{
							Memory.Store(child.symbol, False);
							break;
						}
						}
					}
					else if (node is BinaryNode<bool>)
					{
						BinaryNode<bool> binary            = node as BinaryNode<bool>;
						ScalarSymbolNode<BitVector32> left = binary.GetChildren()[0] as ScalarSymbolNode<BitVector32>;
						LiteralNode<BitVector32> right     = binary.GetChildren()[1] as LiteralNode<BitVector32>;
						if (left != null && right != null && binary.op == "==")
						{
							Memory.Store(left.symbol, right.evaluations[0]);
						}
					}
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
		
		private static void InterpretBasicBlock (Block block)
		{
			Print.DebugMessage(String.Format("Entering basic block with label '{0}'", block.Label), 5);
			// Execute all the statements
			foreach (Cmd cmd in block.Cmds)
			{
				if (cmd is AssignCmd)
				{
					AssignCmd assign = cmd as AssignCmd;
					Print.DebugMessage(assign.ToString(), 5);
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
				}
				if (cmd is CallCmd)
				{
					CallCmd call = cmd as CallCmd;
					if (Regex.IsMatch(call.callee, "_LOG_READ_", RegexOptions.IgnoreCase))
						LogRead(call);
					else if (Regex.IsMatch(call.callee, "_LOG_WRITE_", RegexOptions.IgnoreCase))
						LogWrite(call);
				}
				if (cmd is AssertCmd)
				{
					AssertCmd assert = cmd as AssertCmd;					
					Print.DebugMessage(assert.ToString(), 5);
					ExprTree exprTree = GetExprTree(assert.Expr);
					if (!failedAsserts.Contains(assert))
					{
						EvaluateExprTree(exprTree);
						if (exprTree.evaluation.Equals(False))
						{
							Print.VerboseMessage("Falsifying assertion: " + assert.ToString());
							failedAsserts.Add(assert);
							passedAsserts.Remove(assert);
						}
						else if (!passedAsserts.Contains(assert))
							passedAsserts.Add(assert);
					}
				}
			}
			// Now transfer control
			TransferControl(block);
		}
		
		private static void EvaluateBinaryNode (BinaryNode<BitVector32> binary)
		{
			Print.DebugMessage("Evaluating binary bv node", 5);
			ExprNode<BitVector32> left  = binary.GetChildren()[0] as ExprNode<BitVector32>;
			ExprNode<BitVector32> right = binary.GetChildren()[1] as ExprNode<BitVector32>;
			if (left != null && right != null)
			{
				if (left.evaluations.Count > 0 && right.evaluations.Count > 0)
				{
					foreach (BitVector32 lhs in left.evaluations)
					{
						foreach (BitVector32 rhs in right.evaluations)
						{
							switch (binary.op)
							{
							case "+":
								binary.evaluations.Add(new BitVector32(lhs.Data + rhs.Data));
								break;
							case "-":
								binary.evaluations.Add(new BitVector32(lhs.Data - rhs.Data));
								break;
							case "*":
								binary.evaluations.Add(new BitVector32(lhs.Data * rhs.Data));
								break;
							case "/":
								binary.evaluations.Add(new BitVector32(lhs.Data / rhs.Data));
								break;
							case "BV32_ADD":
								binary.evaluations.Add(new BitVector32(lhs.Data + rhs.Data));
								break;
							case "BV32_SUB":
								binary.evaluations.Add(new BitVector32(lhs.Data - rhs.Data));
								break;
							case "BV32_MUL":
								binary.evaluations.Add(new BitVector32(lhs.Data * rhs.Data));
								break;
							case "BV32_DIV":
								binary.evaluations.Add(new BitVector32(lhs.Data / rhs.Data));
								break;
							case "BV32_AND":
								binary.evaluations.Add(new BitVector32(lhs.Data & rhs.Data));
								break;
							default:
								Print.ExitMessage("Unhandled bv binary op: " + binary.op);
								break;
							}
						}
					}
				}
			}
		}
		
		private static void EvaluateBinaryNode (BinaryNode<bool> binary)
		{
			Print.DebugMessage("Evaluating binary bool node", 5);
			if (binary.op == "||"  ||
			    binary.op == "&&"  ||
			    binary.op == "==>")
			{
				ExprNode<bool> left  = binary.GetChildren()[0] as ExprNode<bool>;
				ExprNode<bool> right = binary.GetChildren()[1] as ExprNode<bool>;
				if (left != null && right != null)
				{
					if (left.evaluations.Count > 0 && right.evaluations.Count > 0)
					{
						foreach (bool lhs in left.evaluations)
						{
							foreach (bool rhs in right.evaluations)
							{
								switch (binary.op)
								{
								case "||":
									binary.evaluations.Add(lhs || rhs);
									break;
								case "&&":
									binary.evaluations.Add(lhs && rhs);
									break;
								case "==>":
									if (lhs && !rhs)
										binary.evaluations.Add(false);
									else
										binary.evaluations.Add(true);
									break;
								default:
									Print.ExitMessage("Unhandled bool binary op: " + binary.op);
									break;
								}
							}
						}
					}
					else
						binary.evaluations.Add(true);
				}
			}
			else
			{
				ExprNode<BitVector32> left  = binary.GetChildren()[0] as ExprNode<BitVector32>;
				ExprNode<BitVector32> right = binary.GetChildren()[1] as ExprNode<BitVector32>;
				if (left != null && right != null)
				{
					foreach (var LevalReval in left.evaluations.Zip(right.evaluations))
					{
						if (left.evaluations.Count > 0 && right.evaluations.Count > 0)
						{
							foreach (BitVector32 lhs in left.evaluations)
							{
								foreach (BitVector32 rhs in right.evaluations)
								{
									switch (binary.op)
									{
									case "<":
										binary.evaluations.Add(lhs.Data < rhs.Data);
										break;
									case "<=":
										binary.evaluations.Add(lhs.Data <= rhs.Data);
										break;
									case ">":
										binary.evaluations.Add(lhs.Data > rhs.Data);
										break;
									case ">=":
										binary.evaluations.Add(lhs.Data >= rhs.Data);
										break;
									case "==":
										binary.evaluations.Add(lhs.Data == rhs.Data);
										break;
									case "!=":
										binary.evaluations.Add(lhs.Data != rhs.Data);
										break;
									case "BV32_SLT":
										binary.evaluations.Add(lhs.Data < rhs.Data);
										break;
									case "BV32_SLE":
										binary.evaluations.Add(lhs.Data <= rhs.Data);
										break;
									case "BV32_SGT":
										binary.evaluations.Add(lhs.Data > rhs.Data);
										break;
									case "BV32_SGE":
										binary.evaluations.Add(lhs.Data >= rhs.Data);
										break;
									default:
										Print.ExitMessage("Unhandled bv binary op: " + binary.op);
										break;
									}
								}
							}
						}
						else
							binary.evaluations.Add(true);
					}
				}
			}
		}
		
		private static void EvaluateNaryNode (NaryNode<BitVector32> nary)
		{
			Print.ExitMessage("Evaluating nary node");
		}
		
		private static void EvaluateNaryNode (NaryNode<bool> nary)
		{		
			Print.ExitMessage("Evaluating nary node");
		}
		
		private static void EvaluateExprTree (ExprTree tree)
		{			
			foreach (HashSet<Node> nodes in tree)
			{
				foreach (Node node in nodes)
				{
					if (node is ScalarSymbolNode<BitVector32>)
					{
						ScalarSymbolNode<BitVector32> scalar = (ScalarSymbolNode<BitVector32>) node;
						if (IsRaceArrayOffsetVariable(scalar.symbol))
						{							
							foreach (BitVector32 offset in Memory.GetRaceArrayOffsets(scalar.symbol))
								scalar.evaluations.Add(offset);
						}
						else
							scalar.evaluations.Add(Memory.GetValue(scalar.symbol));
					}
					else if (node is ScalarSymbolNode<bool>)
					{
						ScalarSymbolNode<bool> scalar = node as ScalarSymbolNode<bool>;
						if (Memory.Contains(scalar.symbol) ||
						    !(tree.Root() is TernaryNode<bool> || tree.Root() is TernaryNode<BitVector32>))
						{
							if (Memory.GetValue(scalar.symbol).Equals(True))
								scalar.evaluations.Add(true);
							else
								scalar.evaluations.Add(false);
						}
					}
					else if (node is MapSymbolNode<BitVector32>)
					{
						MapSymbolNode<BitVector32> map = node as MapSymbolNode<BitVector32>;
						SubscriptExpr subscriptExpr = new SubscriptExpr();
						foreach (ExprNode<BitVector32> child in map.GetChildren())
						{
							BitVector32 subscript = child.evaluations[0];
							subscriptExpr.AddIndex(subscript);
						}
						map.evaluations.Add(Memory.GetValue(map.basename, subscriptExpr));
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
							unary.evaluations.Add(!child.evaluations[0]);
							break;
						}
						}
					}
					else if (node is BinaryNode<BitVector32>)
					{
						EvaluateBinaryNode((BinaryNode<BitVector32>) node);
					}
					else if (node is BinaryNode<bool>)
					{
						EvaluateBinaryNode((BinaryNode<bool>) node);
					}
					else if (node is TernaryNode<bool>)
					{
						TernaryNode<bool> ternary = node as TernaryNode<bool>;
						ExprNode<bool> one   = (ExprNode<bool>) ternary.GetChildren()[0];
						ExprNode<bool> two   = (ExprNode<bool>) ternary.GetChildren()[1];
						ExprNode<bool> three = (ExprNode<bool>) ternary.GetChildren()[2];
						if (one.evaluations[0])
							ternary.evaluations.Add(two.evaluations[0]);
						else
							ternary.evaluations.Add(three.evaluations[0]);
					}
					else if (node is TernaryNode<BitVector32>)
					{
						TernaryNode<BitVector32> ternary = node as TernaryNode<BitVector32>;
						ExprNode<bool> one          = (ExprNode<bool>) ternary.GetChildren()[0];
						ExprNode<BitVector32> two   = (ExprNode<BitVector32>) ternary.GetChildren()[1];
						ExprNode<BitVector32> three = (ExprNode<BitVector32>) ternary.GetChildren()[2];
						if (one.evaluations[0])
							ternary.evaluations.Add(two.evaluations[0]);
						else
							ternary.evaluations.Add(three.evaluations[0]);
					}
					else if (node is NaryNode<BitVector32>)
					{
						EvaluateNaryNode((NaryNode<BitVector32>) node);
					}
					else if (node is NaryNode<bool>)
					{
						EvaluateNaryNode((NaryNode<bool>) node);
					}
				}
			}
			
			Node root = tree.Root();
			if (root is ExprNode<bool>)
			{
				ExprNode<bool> boolRoot = root as ExprNode<bool>;
				tree.evaluation = True;
				foreach (bool eval in boolRoot.evaluations)
				{
					if (!eval)
					{
						tree.evaluation = False;
						break;
					}
				}		
			}
			else
			{
				ExprNode<BitVector32> bvRoot = root as ExprNode<BitVector32>;
				Print.ConditionalExitMessage(bvRoot.evaluations.Count == 1, "Number of bv evaluations should be 1");
				tree.evaluation = bvRoot.evaluations[0];
			}
		}
							
		private static void LogRead (CallCmd call)
		{
			Print.DebugMessage("In log read", 10);
			int dollarIndex = call.callee.IndexOf('$');
			Print.ConditionalExitMessage(dollarIndex >= 0, "Unable to find dollar sign");
			string raceArrayOffsetName = "_READ_OFFSET_" + call.callee.Substring(dollarIndex) + "$1";
			Print.ConditionalExitMessage(Memory.HadRaceArrayVariable(raceArrayOffsetName), "Unable to find array read offset variable: " + raceArrayOffsetName);
			Expr offsetExpr = call.Ins[1];
			ExprTree tree   = GetExprTree(offsetExpr);
			EvaluateExprTree(tree);
			Memory.AddRaceArrayOffset(raceArrayOffsetName, tree.evaluation);
		}
		
		private static void LogWrite (CallCmd call)
		{
			Print.DebugMessage("In log write", 10);
			int dollarIndex = call.callee.IndexOf('$');
			Print.ConditionalExitMessage(dollarIndex >= 0, "Unable to find dollar sign");
			string raceArrayOffsetName = "_WRITE_OFFSET_" + call.callee.Substring(dollarIndex) + "$1";
			Print.ConditionalExitMessage(Memory.HadRaceArrayVariable(raceArrayOffsetName), "Unable to find array read offset variable: " + raceArrayOffsetName);
			Expr offsetExpr = call.Ins[1];
			ExprTree tree   = GetExprTree(offsetExpr);
			EvaluateExprTree(tree);
			Memory.AddRaceArrayOffset(raceArrayOffsetName, tree.evaluation);
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

