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
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Collections.Specialized;
using System.Text.RegularExpressions;
using Microsoft.Boogie;
using Microsoft.Basetypes;
using ConcurrentHoudini = Microsoft.Boogie.Houdini.ConcurrentHoudini;

namespace DynamicAnalysis
{
    class UnhandledException : Exception
    {
        public UnhandledException(string message)
			: base(message)
        { 
        }
    }

    public class BoogieInterpreter
    {
        private Program program;
        private Tuple<int, int, int> threadID;
        private Tuple<int, int, int> groupID;
        private GPU gpu = new GPU();
        private Implementation impl;
        private Block current = null;
        private Random Random = new Random();
        private Memory Memory = new Memory();
        private Dictionary<Expr, ExprTree> ExprTrees = new Dictionary<Expr, ExprTree>();
        private Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
        private Dictionary<AssertCmd, bool> AssertStatus = new Dictionary<AssertCmd, bool>();
        private Dictionary<Tuple<BitVector, BitVector, string>, BitVector> FPBVInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, BitVector>();
        private Dictionary<Tuple<BitVector, BitVector, string>, bool> FPBoolInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, bool>();
        
        public BoogieInterpreter(Program program, Tuple<int, int, int> threadID, Tuple<int, int, int> groupID)
        {
            Console.WriteLine("Falsyifying invariants with dynamic analysis...");
            this.program = program;
            this.threadID = threadID;
            this.groupID = groupID;
            EvaulateAxioms(program.TopLevelDeclarations.OfType<Axiom>());
            EvaluateGlobalVariables(program.TopLevelDeclarations.OfType<GlobalVariable>());
            EvaluateConstants(program.TopLevelDeclarations.OfType<Constant>());			
            InterpretKernels(program.TopLevelDeclarations.OfType<Implementation>().Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "kernel")));
            Console.WriteLine("{0} invariants out of {1} falsified", FalsfifiedInvariants(), TotalInvariants());
        }

        public int FalsfifiedInvariants()
        {
            int count = 0;
            foreach (var tupleKey in AssertStatus)
            {
                if (!tupleKey.Value)
                    count++;
            }
            return count;
        }

        public int TotalInvariants()
        {
            return AssertStatus.Keys.Count;
        }

        private BitVector GetRandomBV(int width)
        {
            if (width == 1)
                return new BitVector(Random.Next(0, 1));
            int lowestVal = 1;
            int highestVal = 16;
            return new BitVector(Random.Next(lowestVal, highestVal+1));
        }

        private bool IsRaceArrayOffsetVariable(string name)
        {
            return Regex.IsMatch(name, "_(WRITE|READ|ATOMIC)_OFFSET_", RegexOptions.IgnoreCase);
        }

        private ExprTree GetExprTree(Expr expr)
        {
            if (!ExprTrees.ContainsKey(expr))
                ExprTrees[expr] = new ExprTree(expr);
            ExprTrees[expr].ClearState();
            return ExprTrees[expr];
        }

        private void EvaluateGlobalVariables(IEnumerable<GlobalVariable> declarations)
        {
            foreach (GlobalVariable decl in declarations)
            {
                if (decl.TypedIdent.Type is MapType)
                    Memory.AddGlobalArray(decl.Name);
                if (IsRaceArrayOffsetVariable(decl.Name))
                    Memory.AddRaceArrayVariable(decl.Name);
            }
        }

        private void EvaluateConstants(IEnumerable<Constant> constants)
        {
            foreach (Constant constant in constants)
            {
                bool existential = false;
                if (constant.CheckBooleanAttribute("existential", ref existential))
                {
                    if (existential)
                        Memory.Store(constant.Name, BitVector.True);
                    else
                        Memory.Store(constant.Name, BitVector.False);
                }
                else if (Regex.IsMatch(constant.Name, "local_id_x", RegexOptions.IgnoreCase))
                {
                    if (threadID.Item1 > -1)
                    {
                        if (threadID.Item1 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.blockDim[DIMENSION.X] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(threadID.Item1));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.blockDim[DIMENSION.X] - 1)));
                }
                else if (Regex.IsMatch(constant.Name, "local_id_y", RegexOptions.IgnoreCase))
                {
                    if (threadID.Item2 > -1)
                    {
                        if (threadID.Item2 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.blockDim[DIMENSION.Y] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(threadID.Item2));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.blockDim[DIMENSION.Y] - 1)));
                }
                else if (Regex.IsMatch(constant.Name, "local_id_z", RegexOptions.IgnoreCase))
                {
                    if (threadID.Item3 > -1)
                    {
                        if (threadID.Item3 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.blockDim[DIMENSION.Z] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(threadID.Item3));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.blockDim[DIMENSION.Z] - 1)));
                }
                else if (Regex.IsMatch(constant.Name, "group_id_x", RegexOptions.IgnoreCase))
                {
                    if (groupID.Item1 > -1)
                    {
                        if (groupID.Item1 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.gridDim[DIMENSION.X] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(groupID.Item1));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.gridDim[DIMENSION.X] - 1)));
                }
                else if (Regex.IsMatch(constant.Name, "group_id_y", RegexOptions.IgnoreCase))
                {
                    if (groupID.Item2 > -1)
                    {
                        if (groupID.Item2 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.gridDim[DIMENSION.Y] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(groupID.Item2));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.gridDim[DIMENSION.Y] - 1)));
                }
                else if (Regex.IsMatch(constant.Name, "group_id_z", RegexOptions.IgnoreCase))
                {
                    if (groupID.Item3 > -1)
                    {
                        if (groupID.Item3 == int.MaxValue)
                            Memory.Store(constant.Name, new BitVector(gpu.gridDim[DIMENSION.Z] - 1));
                        else
                            Memory.Store(constant.Name, new BitVector(groupID.Item3));
                    }
                    else
                        Memory.Store(constant.Name, new BitVector(Random.Next(0, gpu.gridDim[DIMENSION.Z] - 1)));
                }
            }
        }

        private void EvaulateAxioms(IEnumerable<Axiom> axioms)
        {
            foreach (Axiom axiom in axioms)
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
                        BinaryNode<bool> binary = (BinaryNode<bool>)node;
                        if (binary.op == "==")
                        {
                            // Assume that equality is actually assignment into the variable of interest
                            search = false;
                            ScalarSymbolNode<BitVector> left = (ScalarSymbolNode<BitVector>)binary.GetChildren()[0];
                            LiteralNode<BitVector> right = (LiteralNode<BitVector>)binary.GetChildren()[1];
                            if (left.symbol == "group_size_x")
                            {
                                gpu.blockDim[DIMENSION.X] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "group_size_y")
                            {
                                gpu.blockDim[DIMENSION.Y] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "group_size_z")
                            {
                                gpu.blockDim[DIMENSION.Z] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Z]));
                            }
                            else if (left.symbol == "num_groups_x")
                            {
                                gpu.gridDim[DIMENSION.X] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "num_groups_y")
                            {
                                gpu.gridDim[DIMENSION.Y] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "num_groups_z")
                            {
                                gpu.gridDim[DIMENSION.Z] = right.evaluations[0].ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Z]));
                            }
                            else
                                throw new UnhandledException("Unhandled GPU axiom: " + axiom.ToString());
                        }
                    }
                    foreach (Node child in node.GetChildren())
                        stack.Push(child);
                }
            }
        }

        private void InterpretKernels(IEnumerable<Implementation> implementations)
        {
            try
            {
                foreach (Implementation impl in implementations)
                {
                    Print.VerboseMessage(String.Format("Interpreting implementation '{0}'", impl.Name));
                    this.impl = impl;
                    foreach (Requires requires in impl.Proc.Requires)
                    {
                        EvaluateRequires(requires);
                    }
                    foreach (Block block in impl.Blocks)
                    {
                        LabelToBlock[block.Label] = block;
                    }
                    InitialiseFormalParams(impl.InParams);
                    Memory.Dump();
                    current = impl.Blocks[0];
                    while (current != null)
                    {
                        InterpretBasicBlock();
                        current = TransferControl();
                    }
                }
            }
            finally
            {
                Memory.Dump();
                Print.DebugMessage(Output, 10);
            }
        }

        private void EvaluateRequires(Requires requires)
        {
            // The following code currently ignores requires which are implications
            ExprTree tree = new ExprTree(requires.Condition);	
            OpNode<bool> root = tree.Root() as OpNode<bool>;
            if (root != null)
            {          
                foreach (HashSet<Node> nodes in tree)
                {	
                    foreach (Node node in nodes)
                    {
                        if (node is ScalarSymbolNode<bool> && root.op != "==>")
                        {
                            // Initially assume the boolean variable holds. If it is negated this will be handled
                            // further up in the expression tree
                            ScalarSymbolNode<bool> scalar = (ScalarSymbolNode<bool>)node;
                            Memory.Store(scalar.symbol, BitVector.True);
                        }
                        else if (node is UnaryNode<bool> && root.op != "==>")
                        {
                            UnaryNode<bool> unary = node as UnaryNode<bool>;
                            ScalarSymbolNode<bool> child = (ScalarSymbolNode<bool>)unary.GetChildren()[0];
                            switch (unary.op)
                            {
                                case "!":
                                    {
                                        Memory.Store(child.symbol, BitVector.False);
                                        break;
                                    }
                            }
                        }
                        else if (node is BinaryNode<bool>)
                        {
                            BinaryNode<bool> binary = node as BinaryNode<bool>;
                            ScalarSymbolNode<BitVector> left = binary.GetChildren()[0] as ScalarSymbolNode<BitVector>;
                            LiteralNode<BitVector> right = binary.GetChildren()[1] as LiteralNode<BitVector>;
                            if (left != null && right != null && binary.op == "==")
                            {
                                Memory.Store(left.symbol, right.evaluations[0]);
                            }
                        }
                    }
                }
            }
        }

        private void InitialiseFormalParams(List<Variable> formals)
        {
            foreach (Variable v in formals)
            {
                // Only initialise formal parameters not initialised through requires clauses
                if (!Memory.Contains(v.Name))
                {
                    Print.VerboseMessage(String.Format("Formal parameter '{0}' with type '{1}' is uninitialised", v.Name, v.TypedIdent.Type.ToString()));
                    if (v.TypedIdent.Type is BvType)
                    {
                        BvType bv = (BvType)v.TypedIdent.Type;
                        Memory.Store(v.Name, GetRandomBV(bv.Bits));
                    }
                    else if (v.TypedIdent.Type is BasicType)
                    {
                        BasicType basic = (BasicType)v.TypedIdent.Type;
                        if (basic.IsInt)
                            Memory.Store(v.Name, GetRandomBV(32));
                        else
                            throw new UnhandledException(String.Format("Unhandled basic type '{0}'", basic.ToString()));
                    }
                    else
                        throw new UnhandledException("Unknown data type " + v.TypedIdent.Type.ToString());
                }
			}
        }

        private bool IsFormalParameter(string symbol)
        {
            foreach (Variable v in impl.InParams)
            {
                if (symbol.Equals(v.Name))
                    return true;
            }
            return false;
        }

        private void InterpretBasicBlock()
        {
            Print.DebugMessage(String.Format("==========> Entering basic block with label '{0}'", current.Label), 1);
            // Execute all the statements
            foreach (Cmd cmd in current.Cmds)
            {   
                Console.Write(cmd.ToString());
                
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
                            MapAssignLhs lhs = (MapAssignLhs)LhsEval.Item1;
                            SubscriptExpr subscriptExpr = new SubscriptExpr();
                            foreach (Expr index in lhs.Indexes)
                            {
                                ExprTree exprTree = GetExprTree(index);
                                EvaluateExprTree(exprTree);
                                BitVector subscript = exprTree.evaluation;
                                subscriptExpr.AddIndex(subscript);
                            }
                            ExprTree tree = LhsEval.Item2;
                            if (!tree.unitialised)
                                Memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree.evaluation);
                        }
                        else
                        {
                            SimpleAssignLhs lhs = (SimpleAssignLhs)LhsEval.Item1;
                            ExprTree tree = LhsEval.Item2;
                            if (!tree.unitialised)
                                Memory.Store(lhs.AssignedVariable.Name, tree.evaluation);
                        }
                    }
                }
                else if (cmd is CallCmd)
                {
                    CallCmd call = cmd as CallCmd;
                    if (Regex.IsMatch(call.callee, "_LOG_READ_", RegexOptions.IgnoreCase))
                        LogRead(call);
                    else if (Regex.IsMatch(call.callee, "_LOG_WRITE_", RegexOptions.IgnoreCase))
                        LogWrite(call);
                    else if (Regex.IsMatch(call.callee, "bugle_barrier", RegexOptions.IgnoreCase))
                        Barrier(call);
                }
                else if (cmd is AssertCmd)
                {
                    AssertCmd assert = cmd as AssertCmd;
                    // Only check asserts which have attributes as these are the conjectured invariants
                    if (assert.Attributes != null)
                    {
                        ExprTree exprTree = GetExprTree(assert.Expr);
                        if (!AssertStatus.ContainsKey(assert))
                            AssertStatus[assert] = true;
                        if (AssertStatus[assert])
                        {
                            EvaluateExprTree(exprTree);
                            if (exprTree.evaluation.Equals(BitVector.False))
                            {
                                AssertStatus[assert] = false;
                                Regex r = new Regex("_[a-z][0-9]+");
                                MatchCollection matches = r.Matches(assert.ToString());
                                string BoogieVariable = null;
                                foreach (Match match in matches)
                                {
                                    foreach (Capture capture in match.Captures)
                                    {
                                        BoogieVariable = capture.Value;
                                    }
                                }
                                Print.ConditionalExitMessage(BoogieVariable != null, "Unable to find Boogie variable");        
                                Console.Write("Removing " + assert.ToString());
                                ConcurrentHoudini.RefutedAnnotation annotation = GPUVerify.GVUtil.getRefutedAnnotation(program, BoogieVariable, impl.Name);
                                ConcurrentHoudini.RefutedSharedAnnotations[BoogieVariable] = annotation;
                            }
                        }
                    }
                }
                else if (cmd is HavocCmd)
                {
                    HavocCmd havoc = cmd as HavocCmd;
                    foreach (IdentifierExpr id in havoc.Vars)
                    {
                        if (id.Type is BvType)
                        {
                            BvType bv = (BvType)id.Type;
                            Memory.Store(id.Name, GetRandomBV(bv.Bits));
                        }
                    }
                }
                else if (cmd is AssumeCmd)
                {
                    AssumeCmd assume = cmd as AssumeCmd;
                }
                else
                    throw new UnhandledException("Unhandled command: " + cmd.ToString());
            }
        }

        private void EvaluateBinaryNode(BinaryNode<BitVector> binary)
        {
            Print.DebugMessage("Evaluating binary bv node", 10);
            ExprNode<BitVector> left = binary.GetChildren()[0] as ExprNode<BitVector>;
            ExprNode<BitVector> right = binary.GetChildren()[1] as ExprNode<BitVector>;
            if (left != null && right != null)
            {
                if (left.evaluations.Count > 0 && right.evaluations.Count > 0)
                {
                    foreach (BitVector lhs in left.evaluations)
                    {
                        foreach (BitVector rhs in right.evaluations)
                        {
                            switch (binary.op)
                            {
                                case "+":
                                    binary.evaluations.Add(lhs + rhs);
                                    break;
                                case "-":
                                    binary.evaluations.Add(lhs - rhs);
                                    break;
                                case "*":
                                    binary.evaluations.Add(lhs * rhs);
                                    break;
                                case "/":
                                    binary.evaluations.Add(lhs / rhs);
                                    break;
                                case "BV32_UREM":
                                    {
                                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                        binary.evaluations.Add(lhsUnsigned % rhsUnsigned);
                                        break;
                                    }
                                case "BV32_UDIV":
                                    {
                                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                        binary.evaluations.Add(lhsUnsigned / rhsUnsigned);
                                        break;
                                    }
                                case "BV32_SDIV":
                                    binary.evaluations.Add(lhs / rhs);
                                    break;
                                case "BV32_LSHR":
                                    binary.evaluations.Add(lhs >> rhs.ConvertToInt32());
                                    break;
                                case "BV32_SHL":
                                    binary.evaluations.Add(lhs << rhs.ConvertToInt32());
                                    break;
                                case "BV32_ADD":
                                    binary.evaluations.Add(lhs + rhs);
                                    break;
                                case "BV32_SUB":
                                    binary.evaluations.Add(lhs - rhs);
                                    break;
                                case "BV32_MUL":
                                    binary.evaluations.Add(lhs * rhs);
                                    break;
                                case "BV32_DIV":
                                    binary.evaluations.Add(lhs / rhs);
                                    break;
                                case "BV32_AND":
                                    binary.evaluations.Add(lhs & rhs);
                                    break;
                                case "FADD32":
                                case "FSUB32":
                                case "FMUL32":
                                case "FDIV32":
                                    {
                                        Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(lhs, rhs, binary.op);
                                        if (!FPBVInterpretations.ContainsKey(FPTriple))
                                            FPBVInterpretations[FPTriple] = new BitVector(Random.Next());
                                        binary.evaluations.Add(FPBVInterpretations[FPTriple]);
                                        break;
                                    }
                                case "FADD64":
                                case "FSUB64":
                                case "FMUL64":
                                case "FDIV64":
                                    {
                                        Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(lhs, rhs, binary.op);
                                        if (!FPBVInterpretations.ContainsKey(FPTriple))
                                            FPBVInterpretations[FPTriple] = new BitVector(Random.Next());
                                        binary.evaluations.Add(FPBVInterpretations[FPTriple]);
                                        break;
                                    }
                                default:
                                    throw new UnhandledException("Unhandled bv binary op: " + binary.op);
                            }
                        }
                    }
                }
            }
        }

        private void EvaluateBinaryNode(BinaryNode<bool> binary)
        {
            Print.DebugMessage("Evaluating binary bool node", 10);
            if (binary.op == "||" ||
                binary.op == "&&" ||
                binary.op == "==>")
            {
                ExprNode<bool> left = binary.GetChildren()[0] as ExprNode<bool>;
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
                                        throw new UnhandledException("Unhandled bool binary op: " + binary.op);
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
                ExprNode<BitVector> left = binary.GetChildren()[0] as ExprNode<BitVector>;
                ExprNode<BitVector> right = binary.GetChildren()[1] as ExprNode<BitVector>;
                if (left != null && right != null)
                {
                    foreach (var LevalReval in left.evaluations.Zip(right.evaluations))
                    {
                        if (left.evaluations.Count > 0 && right.evaluations.Count > 0)
                        {
                            foreach (BitVector lhs in left.evaluations)
                            {
                                foreach (BitVector rhs in right.evaluations)
                                {
                                    switch (binary.op)
                                    {
                                        case "<":
                                            binary.evaluations.Add(lhs < rhs);
                                            break;
                                        case "<=":
                                            binary.evaluations.Add(lhs <= rhs);
                                            break;
                                        case ">":
                                            binary.evaluations.Add(lhs > rhs);
                                            break;
                                        case ">=":
                                            binary.evaluations.Add(lhs >= rhs);
                                            break;
                                        case "==":
                                            binary.evaluations.Add(lhs == rhs);
                                            break;
                                        case "!=":
                                            binary.evaluations.Add(lhs != rhs);
                                            break;
                                        case "BV32_ULT":
                                            {
                                                BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                                BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                                binary.evaluations.Add(lhsUnsigned < rhsUnsigned);
                                                break;
                                            }
                                        case "BV32_ULE":
                                            {
                                                BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                                BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                                binary.evaluations.Add(lhsUnsigned <= rhsUnsigned);
                                                break;
                                            }
                                        case "BV32_UGT":
                                            {
                                                BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                                BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                                binary.evaluations.Add(lhsUnsigned > rhsUnsigned);
                                                break;
                                            }
                                        case "BV32_UGE":
                                            {
                                                BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                                                BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                                                binary.evaluations.Add(lhsUnsigned >= rhsUnsigned);
                                                break;
                                            }
                                        case "BV32_SLT":
                                            binary.evaluations.Add(lhs < rhs);
                                            break;
                                        case "BV32_SLE":
                                            binary.evaluations.Add(lhs <= rhs);
                                            break;
                                        case "BV32_SGT":
                                            binary.evaluations.Add(lhs > rhs);
                                            break;
                                        case "BV32_SGE":
                                            binary.evaluations.Add(lhs >= rhs);
                                            break;
                                        case "FLT32":
                                        case "FLE32":
                                        case "FGT32":
                                        case "FGE32":
                                        case "FLT64":
                                        case "FLE64":
                                        case "FGT64":
                                        case "FGE64":
                                            {
                                                Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(lhs, rhs, binary.op);
                                                if (!FPBoolInterpretations.ContainsKey(FPTriple))
                                                {
                                                    if (Random.Next(0, 2) == 0)
                                                        FPBoolInterpretations[FPTriple] = false;
                                                    else
                                                        FPBoolInterpretations[FPTriple] = true;
                                                }
                                                binary.evaluations.Add(FPBoolInterpretations[FPTriple]);
                                                break;
                                            }
                                        default:
                                            throw new UnhandledException("Unhandled bool bv binary op: " + binary.op);
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

        private void EvaluateExprTree(ExprTree tree)
        {			
            foreach (HashSet<Node> nodes in tree)
            {
                foreach (Node node in nodes)
                {
                    if (node.IsLeaf())
                    {
                        if (node is ScalarSymbolNode<BitVector>)
                        {
                            ScalarSymbolNode<BitVector> scalar = (ScalarSymbolNode<BitVector>)node;
                            if (IsRaceArrayOffsetVariable(scalar.symbol))
                            {							
                                foreach (BitVector offset in Memory.GetRaceArrayOffsets(scalar.symbol))
                                    scalar.evaluations.Add(offset);
                            }
                            else
                            {
                                if (!Memory.Contains(scalar.symbol))
                                {
                                    scalar.uninitialised = true;
                                }
                                else
                                    scalar.evaluations.Add(Memory.GetValue(scalar.symbol));
                            }
                        }
                        else if (node is ScalarSymbolNode<bool>)
                        {
                            ScalarSymbolNode<bool> scalar = node as ScalarSymbolNode<bool>;
                            if (!Memory.Contains(scalar.symbol))
                            {
                                scalar.uninitialised = true;
                            }
                            else
                            {
                                if (Memory.GetValue(scalar.symbol).Equals(BitVector.True))
                                    scalar.evaluations.Add(true);
                                else
                                    scalar.evaluations.Add(false);   
                            }
                        }
                    }
                    else
                    {
                        if (node is MapSymbolNode<BitVector>)
                        {
                            MapSymbolNode<BitVector> map = node as MapSymbolNode<BitVector>;
                            SubscriptExpr subscriptExpr = new SubscriptExpr();
                            foreach (ExprNode<BitVector> child in map.GetChildren())
                            {
                                BitVector subscript = child.evaluations[0];
                                subscriptExpr.AddIndex(subscript);
                            }
                            map.evaluations.Add(Memory.GetValue(map.basename, subscriptExpr));
                        }
                        else if (node is MapSymbolNode<bool>)
                        {
                            MapSymbolNode<bool> map = node as MapSymbolNode<bool>;
                            throw new UnhandledException("Unhandled map expression: " + map.ToString());
                        }
                        else if (node is UnaryNode<bool>)
                        {
                            UnaryNode<bool> unary = node as UnaryNode<bool>;
                            ExprNode<bool> child = (ExprNode<bool>)unary.GetChildren()[0];
                            switch (unary.op)
                            {
                                case "!":
                                    if (child.uninitialised)
                                        node.uninitialised = true;
                                    else
                                        unary.evaluations.Add(!child.evaluations[0]);
                                    break;
                            }
                        }
                        else if (node is UnaryNode<BitVector>)
                        {
                            UnaryNode<BitVector> unary = node as UnaryNode<BitVector>;
                            ExprNode<BitVector> child = (ExprNode<BitVector>)unary.GetChildren()[0];
                            switch (unary.op)
                            {
                                case "FSQRT32":
                                case "FSQRT64":
                                case "FLOG32":
                                case "FLOG64":
                                case "FABS32":
                                case "FABS64":
                                case "FEXP32":
                                case "FEXP64":
                                    {
                                        Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(child.evaluations[0], BitVector.Zero, unary.op);
                                        if (!FPBVInterpretations.ContainsKey(FPTriple))
                                            FPBVInterpretations[FPTriple] = new BitVector(Random.Next());
                                        unary.evaluations.Add(FPBVInterpretations[FPTriple]);
                                        break;
                                    }
                                case "BV1_ZEXT32":
                                    BitVector ZeroExtended = BitVector.ZeroExtend(child.evaluations[0], 31);
                                    unary.evaluations.Add(ZeroExtended);
                                    break;
                                case "UI32_TO_FP32":
                                case "SI32_TO_FP32":
                                    unary.evaluations.Add(child.evaluations[0]);
                                    break;
                                default:
                                    throw new UnhandledException("Unhandled bv unary op: " + unary.op);
                            }
                        }
                        else if (node is BinaryNode<BitVector>)
                        {
                            BinaryNode<BitVector> binary = (BinaryNode<BitVector>)node;
                            EvaluateBinaryNode(binary);
                            if (binary.evaluations.Count == 0)
                                binary.uninitialised = true;
                        }
                        else if (node is BinaryNode<bool>)
                        {                     
                            BinaryNode<bool> binary = (BinaryNode<bool>)node;
                            EvaluateBinaryNode(binary);
                            if (binary.evaluations.Count == 0)
                                binary.uninitialised = true;
                        }
                        else if (node is TernaryNode<bool>)
                        {
                            TernaryNode<bool> ternary = node as TernaryNode<bool>;
                            ExprNode<bool> one = (ExprNode<bool>)ternary.GetChildren()[0];
                            ExprNode<bool> two = (ExprNode<bool>)ternary.GetChildren()[1];
                            ExprNode<bool> three = (ExprNode<bool>)ternary.GetChildren()[2];
                            if (one.evaluations[0])
                            {
                                if (two.uninitialised)
                                    node.uninitialised = true;
                                else
                                    ternary.evaluations.Add(two.evaluations[0]);
                            }
                            else
                            {
                                if (three.uninitialised)
                                    node.uninitialised = true;
                                else
                                    ternary.evaluations.Add(three.evaluations[0]);
                            }
                        }
                        else if (node is TernaryNode<BitVector>)
                        {
                            TernaryNode<BitVector> ternary = node as TernaryNode<BitVector>;
                            ExprNode<bool> one = (ExprNode<bool>)ternary.GetChildren()[0];
                            ExprNode<BitVector> two = (ExprNode<BitVector>)ternary.GetChildren()[1];
                            ExprNode<BitVector> three = (ExprNode<BitVector>)ternary.GetChildren()[2];
                            if (one.evaluations[0])
                            {
                                if (two.uninitialised)
                                    node.uninitialised = true;
                                else
                                    ternary.evaluations.Add(two.evaluations[0]);
                            }
                            else
                            {
                                if (three.uninitialised)
                                    node.uninitialised = true;
                                else
                                    ternary.evaluations.Add(three.evaluations[0]);
                            }
                        }
                    }
                }
            }
            
            Node root = tree.Root();
            tree.unitialised = root.uninitialised;
            if (root is ExprNode<bool>)
            {
                ExprNode<bool> boolRoot = root as ExprNode<bool>;
                tree.evaluation = BitVector.True;
                foreach (bool eval in boolRoot.evaluations)
                {
                    if (!eval)
                    {
                        tree.evaluation = BitVector.False;
                        break;
                    }
                }       
            }
            else if (!root.uninitialised)
            {
                ExprNode<BitVector> bvRoot = root as ExprNode<BitVector>;
                Print.ConditionalExitMessage(bvRoot.evaluations.Count == 1, "Number of bv evaluations should be 1. Instead found " + bvRoot.evaluations.Count.ToString());
                tree.evaluation = bvRoot.evaluations[0];
            }
        }

        private void Barrier(CallCmd call)
        {
            foreach (string name in Memory.GetRaceArrayVariables())
            {
                if (Memory.GetRaceArrayOffsets(name).Count > 0)
                {
                    int dollarIndex = name.IndexOf('$');
                    Print.ConditionalExitMessage(dollarIndex >= 0, "Unable to find dollar sign");
                    string arrayName = name.Substring(dollarIndex);
                    string accessType = name.Substring(0, dollarIndex);
                    switch (accessType)
                    {
                        case "_WRITE_OFFSET_":
                            {
                                string accessTracker = "_WRITE_HAS_OCCURRED_" + arrayName; 
                                Memory.Store(accessTracker, BitVector.True);
                                break;   
                            }
                        case "_READ_OFFSET_":
                            {
                                string accessTracker = "_READ_HAS_OCCURRED_" + arrayName; 
                                Memory.Store(accessTracker, BitVector.True);
                                break;
                            } 
                        case "_ATOMIC_OFFSET_":
                            {
                                string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName; 
                                Memory.Store(accessTracker, BitVector.True);
                                break;
                            }
                    }
                    
                    
                    Console.WriteLine(name +  " " + arrayName + " " + accessType);
                }
                Memory.ClearRaceArrayOffset(name);
            }
        }

        private void LogRead(CallCmd call)
        {
            Print.DebugMessage("In log read", 10);
            int dollarIndex = call.callee.IndexOf('$');
            Print.ConditionalExitMessage(dollarIndex >= 0, "Unable to find dollar sign");
            string raceArrayOffsetName = "_READ_OFFSET_" + call.callee.Substring(dollarIndex) + "$1";
            Print.ConditionalExitMessage(Memory.HadRaceArrayVariable(raceArrayOffsetName), "Unable to find array read offset variable: " + raceArrayOffsetName);
            Expr offsetExpr = call.Ins[1];
            ExprTree tree = GetExprTree(offsetExpr);
            EvaluateExprTree(tree);
            if (!tree.unitialised)
                Memory.AddRaceArrayOffset(raceArrayOffsetName, tree.evaluation);
        }

        private void LogWrite(CallCmd call)
        {
            Print.DebugMessage("In log write", 10);
            int dollarIndex = call.callee.IndexOf('$');
            Print.ConditionalExitMessage(dollarIndex >= 0, "Unable to find dollar sign");
            string raceArrayOffsetName = "_WRITE_OFFSET_" + call.callee.Substring(dollarIndex) + "$1";
            Print.ConditionalExitMessage(Memory.HadRaceArrayVariable(raceArrayOffsetName), "Unable to find array read offset variable: " + raceArrayOffsetName);
            Expr offsetExpr = call.Ins[1];
            ExprTree tree = GetExprTree(offsetExpr);
            EvaluateExprTree(tree);
            if (!tree.unitialised)
                Memory.AddRaceArrayOffset(raceArrayOffsetName, tree.evaluation);
        }

        private Block TransferControl()
        {
            TransferCmd transfer = current.TransferCmd;
            if (transfer is GotoCmd)
            {
                GotoCmd goto_ = transfer as GotoCmd;
                if (goto_.labelNames.Count == 1)
                {
                    string succLabel = goto_.labelNames[0];
                    Block succ = LabelToBlock[succLabel];
                    return succ;
                }
                else
                {
                    // Loop through all potential successors and find one whose guard evaluates to true
                    foreach (string succLabel in goto_.labelNames)
                    {
                        Block succ = LabelToBlock[succLabel];
                        PredicateCmd predicateCmd = (PredicateCmd)succ.Cmds[0];
                        ExprTree exprTree = GetExprTree(predicateCmd.Expr);
                        EvaluateExprTree(exprTree);
                        if (exprTree.evaluation.Equals(BitVector.True))
                            return succ;
                    }
                    throw new UnhandledException("No successor guard evaluates to true");
                }
            }
            else if (transfer is ReturnCmd)
                return null;
            throw new UnhandledException("Unhandled control transfer command: " + transfer.ToString());
        }

        private void Output()
        {		
            Console.WriteLine("************************ The following asserts do NOT hold ************************");
            foreach (var tupleKey in AssertStatus)
            {
                if (!tupleKey.Value)
                    Console.WriteLine(tupleKey.Key.ToString());
            }
            Console.WriteLine("************************ The following asserts do HOLD ************************");
            foreach (var tupleKey in AssertStatus)
            {
                if (tupleKey.Value)
                    Console.WriteLine(tupleKey.Key.ToString());
            }
        }
    }
}

