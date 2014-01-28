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
using System.Diagnostics.Contracts;
using System.Collections.Specialized;
using System.Text.RegularExpressions;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;
using Microsoft.Basetypes;
using ConcurrentHoudini = Microsoft.Boogie.Houdini.ConcurrentHoudini;

namespace GPUVerify
{
    class UnhandledException : Exception
    {
        public UnhandledException(string message)
         : base(message)
        { 
        }
    }

    internal static class BinaryOps
    {
        public static string OR = "||";
        public static string AND = "&&";
        public static string IF = "==>";
        public static string IFF = "<==>";
        public static string GT = ">";
        public static string GTE = ">=";
        public static string LT = "<";
        public static string LTE = "<=";
        public static string ADD = "+";
        public static string SUBTRACT = "-";
        public static string MULTIPLY = "*";
        public static string DIVIDE = "/";
        public static string NEQ = "!=";
        public static string EQ = "==";
    }

    internal static class RegularExpressions
    {
        public static Regex INVARIANT_VARIABLE = new Regex("_[a-z][0-9]+");
        // Case sensitive
        public static Regex WATCHDOG_VARIABLE = new Regex("_WATCHED_OFFSET", RegexOptions.IgnoreCase);
        public static Regex OFFSET_VARIABLE = new Regex("_(WRITE|READ|ATOMIC)_OFFSET_", RegexOptions.IgnoreCase);
        public static Regex TRACKING_VARIABLE = new Regex("_(WRITE|READ|ATOMIC)_HAS_OCCURRED_", RegexOptions.IgnoreCase);
        public static Regex LOG_READ = new Regex("_LOG_READ_", RegexOptions.IgnoreCase);
        public static Regex LOG_WRITE = new Regex("_LOG_WRITE_", RegexOptions.IgnoreCase);
        public static Regex LOG_ATOMIC = new Regex("_LOG_ATOMIC_", RegexOptions.IgnoreCase);
        public static Regex BUGLE_BARRIER = new Regex("bugle_barrier", RegexOptions.IgnoreCase);
        public static Regex BVSLE = new Regex("BV[0-9]+_SLE", RegexOptions.IgnoreCase);
        public static Regex BVSLT = new Regex("BV[0-9]+_SLT", RegexOptions.IgnoreCase);
        public static Regex BVSGE = new Regex("BV[0-9]+_SGE", RegexOptions.IgnoreCase);
        public static Regex BVSGT = new Regex("BV[0-9]+_SGT", RegexOptions.IgnoreCase);
        public static Regex BVULE = new Regex("BV[0-9]+_ULE", RegexOptions.IgnoreCase);
        public static Regex BVULT = new Regex("BV[0-9]+_ULT", RegexOptions.IgnoreCase);
        public static Regex BVUGE = new Regex("BV[0-9]+_UGE", RegexOptions.IgnoreCase);
        public static Regex BVUGT = new Regex("BV[0-9]+_UGT", RegexOptions.IgnoreCase);
        public static Regex BVASHR = new Regex("BV[0-9]+_ASHR", RegexOptions.IgnoreCase);
        public static Regex BVLSHR = new Regex("BV[0-9]+_LSHR", RegexOptions.IgnoreCase);
        public static Regex BVSHL = new Regex("BV[0-9]+_SHL", RegexOptions.IgnoreCase);
        public static Regex BVADD = new Regex("BV[0-9]+_ADD", RegexOptions.IgnoreCase);
        public static Regex BVSUB = new Regex("BV[0-9]+_SUB", RegexOptions.IgnoreCase);
        public static Regex BVMUL = new Regex("BV[0-9]+_MUL", RegexOptions.IgnoreCase);
        public static Regex BVDIV = new Regex("BV[0-9]+_DIV", RegexOptions.IgnoreCase);
        public static Regex BVAND = new Regex("BV[0-9]+_AND", RegexOptions.IgnoreCase);
        public static Regex BVOR = new Regex("BV[0-9]+_OR", RegexOptions.IgnoreCase);
        public static Regex BVXOR = new Regex("BV[0-9]+_XOR", RegexOptions.IgnoreCase);
        public static Regex BVSREM = new Regex("BV[0-9]+_SREM", RegexOptions.IgnoreCase);
        public static Regex BVUREM = new Regex("BV[0-9]+_UREM", RegexOptions.IgnoreCase);
        public static Regex BVSDIV = new Regex("BV[0-9]+_SDIV", RegexOptions.IgnoreCase);
        public static Regex BVUDIV = new Regex("BV[0-9]+_UDIV", RegexOptions.IgnoreCase);
        public static Regex BVZEXT = new Regex("BV[0-9]+_ZEXT", RegexOptions.IgnoreCase);
        public static Regex BVSEXT = new Regex("BV[0-9]+_SEXT", RegexOptions.IgnoreCase);
        public static Regex CAST_TO_FP = new Regex("(U|S)I[0-9]+_TO_FP[0-9]+", RegexOptions.IgnoreCase);
        public static Regex CAST_TO_INT = new Regex("FP[0-9]+_TO_(U|S)I[0-9]+", RegexOptions.IgnoreCase);
        public static Regex CAST_FP_TO_DOUBLE = new Regex("FP[0-9]+_CONV[0-9]+", RegexOptions.IgnoreCase);
    }

    public class BoogieInterpreter
    {
        private BitVector[] LocalID1 = new BitVector[3];
        private BitVector[] LocalID2 = new BitVector[3];
        private BitVector[] GlobalID1 = new BitVector[3];
        private BitVector[] GlobalID2 = new BitVector[3];
        private GPU gpu = new GPU();
        private Memory Memory = new Memory();
        private Dictionary<Expr, ExprTree> ExprTrees = new Dictionary<Expr, ExprTree>();
        private Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
        private Dictionary<AssertCmd, BitVector> AssertStatus = new Dictionary<AssertCmd, BitVector>();
        private HashSet<string> KilledAsserts = new HashSet<string>();
        private Dictionary<Tuple<BitVector, BitVector, string>, BitVector> FPInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, BitVector>();
        private HashSet<Block> Covered = new HashSet<Block>();
        private int GlobalHeaderCount = 0;
        private Dictionary<Block, int> HeaderExecutionCounts = new Dictionary<Block, int>();
        private Dictionary<Block, List<Block>> HeaderToLoopExitBlocks = new Dictionary<Block, List<Block>>();
        private Dictionary<Block, HashSet<Block>> HeaderToLoopBody = new Dictionary<Block, HashSet<Block>>();
        private int Executions = 0;
        private Random Random;

        public static void Start(Program program, Tuple<int, int, int> threadID, Tuple<int, int, int> groupID)
        {
            Stopwatch timer = new Stopwatch();
            timer.Start();            
            new BoogieInterpreter(program, threadID, groupID);
            timer.Stop();
            Print.VerboseMessage("Dynamic analysis consumed " + timer.Elapsed);
        }

        public BoogieInterpreter(Program program, Tuple<int, int, int> localIDSpecification, Tuple<int, int, int> globalIDSpecification)
        {
            // If there are no invariants to falsify, return
            if (program.TopLevelDeclarations.OfType<Constant>().Where(item => QKeyValue.FindBoolAttribute(item.Attributes, "existential")).Count() == 0)
                return;
                           
            Implementation impl = program.TopLevelDeclarations.OfType<Implementation>().Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "kernel")).First();
            // Seed the random number generator so that it is deterministic
            Random = new Random(impl.Name.Length);
   
            // Build map from label to basic block
            foreach (Block block in impl.Blocks)
                LabelToBlock[block.Label] = block;   

            // Compute loop-exit edges of each natural loop
            Graph<Block> loopInfo = program.ProcessLoops(impl);
            foreach (Block header in loopInfo.Headers)
            {
                HeaderToLoopBody[header] = new HashSet<Block>();
                foreach (Block tail in loopInfo.BackEdgeNodes(header))
                    HeaderToLoopBody[header].UnionWith(loopInfo.NaturalLoops(header, tail));
                ComputeLoopExitBlocks(header, HeaderToLoopBody[header]);
            }

            Print.VerboseMessage("Falsyifying invariants with dynamic analysis...");
            try
            {  
                do
                {
                    // Reset the memory in readiness for the next execution
                    Memory.Clear();
                    EvaulateAxioms(program.TopLevelDeclarations.OfType<Axiom>());
                    EvaluateGlobalVariables(program.TopLevelDeclarations.OfType<GlobalVariable>());
                    Print.DebugMessage(gpu.ToString(), 1);
                    // Set the local thread IDs and group IDs
                    SetLocalIDs(localIDSpecification);
                    SetGlobalIDs(globalIDSpecification);
                    Print.DebugMessage("Thread 1 local  ID = " + String.Join(", ", new List<BitVector>(LocalID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 1 global ID = " + String.Join(", ", new List<BitVector>(GlobalID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 local  ID = " + String.Join(", ", new List<BitVector>(LocalID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 global ID = " + String.Join(", ", new List<BitVector>(GlobalID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    EvaluateConstants(program.TopLevelDeclarations.OfType<Constant>());  
                    InterpretKernel(program, impl, loopInfo.Headers);
                    Executions++;
                } while (GlobalHeaderCount < ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopHeaderLimit
                && !AllBlocksCovered(impl)
                && Executions < 5);
                // The condition states: try to kill invariants while we have not exhausted a global loop header limit 
                // AND not every basic block has been covered
                // AND the number of re-invocations of the kernel does not exceed 5
            }
            finally
            {
                SummarizeKilledInvariants();
                Print.VerboseMessage("Dynamic analysis done");
            }
        }

        private void ComputeLoopExitBlocks(Block header, HashSet<Block> loopBody)
        {
            HeaderToLoopExitBlocks[header] = new List<Block>();
            foreach (Block block in loopBody)
            {
                TransferCmd transfer = block.TransferCmd;
                if (transfer is GotoCmd)
                {
                    GotoCmd goto_ = transfer as GotoCmd;
                    if (goto_.labelNames.Count == 1)
                    {
                        string succLabel = goto_.labelNames[0];
                        Block succ = LabelToBlock[succLabel];
                        if (!loopBody.Contains(succ))
                            HeaderToLoopExitBlocks[header].Add(succ);
                    }
                    else
                    {
                        foreach (string succLabel in goto_.labelNames)
                        {
                            Block succ = LabelToBlock[succLabel];
                            if (!loopBody.Contains(succ))
                                HeaderToLoopExitBlocks[header].Add(succ);
                        }
                    }
                }
            }
        }

        private bool AllBlocksCovered(Implementation impl)
        {
            foreach (Block block in impl.Blocks)
            {
                if (!Covered.Contains(block))
                    return false;
            }
            Print.VerboseMessage("All basic blocks covered");
            return true;
        }

        private void SummarizeKilledInvariants()
        {
            if (KilledAsserts.Count > 0)
            {
                Print.VerboseMessage("Dynamic analysis removed " + KilledAsserts.Count.ToString() + " invariants:");
                foreach (string BoogieVariable in KilledAsserts)
                    Print.VerboseMessage(BoogieVariable);
            }
        }

        private Tuple<BitVector, BitVector> GetID(int selectedValue, int dimensionUpperBound)
        {
            if (selectedValue > -1)
            {
                if (selectedValue == int.MaxValue)
                {
                    BitVector val1 = new BitVector(dimensionUpperBound);
                    BitVector val2;
                    if (dimensionUpperBound > 0)
                        val2 = new BitVector(dimensionUpperBound - 1);
                    else
                        val2 = new BitVector(dimensionUpperBound);
                    return Tuple.Create(val1, val2);
                }
                else
                {
                    BitVector val1 = new BitVector(selectedValue);
                    BitVector val2;
                    if (selectedValue < dimensionUpperBound)
                        val2 = new BitVector(selectedValue + 1);
                    else
                        val2 = new BitVector(selectedValue);
                    return Tuple.Create(val1, val2);
                }
            }
            else
            {
                BitVector val1 = new BitVector(Random.Next(0, dimensionUpperBound + 1));
                BitVector val2 = new BitVector(Random.Next(0, dimensionUpperBound + 1));
                return Tuple.Create(val1, val2);
            }
        }

        private void SetLocalIDs(Tuple<int, int, int> localIDSpecification)
        {
            Tuple<BitVector,BitVector> dimX = GetID(localIDSpecification.Item1, gpu.blockDim[DIMENSION.X] - 1);
            Tuple<BitVector,BitVector> dimY = GetID(localIDSpecification.Item2, gpu.blockDim[DIMENSION.Y] - 1);
            Tuple<BitVector,BitVector> dimZ = GetID(localIDSpecification.Item3, gpu.blockDim[DIMENSION.Z] - 1);
            LocalID1[0] = dimX.Item1;
            LocalID2[0] = dimX.Item2;
            LocalID1[1] = dimY.Item1;
            LocalID2[1] = dimY.Item2;
            LocalID1[2] = dimZ.Item1;
            LocalID2[2] = dimZ.Item2;    
        }

        private void SetGlobalIDs(Tuple<int, int, int> globalIDSpecification)
        {
            Tuple<BitVector,BitVector> dimX = GetID(globalIDSpecification.Item1, gpu.gridDim[DIMENSION.X] - 1);
            Tuple<BitVector,BitVector> dimY = GetID(globalIDSpecification.Item2, gpu.gridDim[DIMENSION.Y] - 1);
            Tuple<BitVector,BitVector> dimZ = GetID(globalIDSpecification.Item3, gpu.gridDim[DIMENSION.Z] - 1);            
            GlobalID1[0] = dimX.Item1;
            GlobalID2[0] = dimX.Item2;
            GlobalID1[1] = dimY.Item1;
            GlobalID2[1] = dimY.Item2;
            GlobalID1[2] = dimZ.Item1;
            GlobalID2[2] = dimZ.Item2;    
        }

        private ExprTree GetExprTree(Expr expr)
        {
            if (!ExprTrees.ContainsKey(expr))
                ExprTrees[expr] = new ExprTree(expr);
            ExprTrees[expr].ClearState();
            return ExprTrees[expr];
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
                    if (node is BinaryNode<BitVector>)
                    {
                        BinaryNode<BitVector> binary = (BinaryNode<BitVector>)node;
                        if (binary.op == "==")
                        {
                            // Assume that equality is actually assignment into the variable of interest
                            search = false;
                            ScalarSymbolNode<BitVector> left = (ScalarSymbolNode<BitVector>)binary.GetChildren()[0];
                            LiteralNode<BitVector> right = (LiteralNode<BitVector>)binary.GetChildren()[1];
                            if (left.symbol == "group_size_x")
                            {
                                gpu.blockDim[DIMENSION.X] = right.GetUniqueElement().ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "group_size_y")
                            {
                                gpu.blockDim[DIMENSION.Y] = right.GetUniqueElement().ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "group_size_z")
                            {
                                gpu.blockDim[DIMENSION.Z] = right.GetUniqueElement().ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Z]));
                            }
                            else if (left.symbol == "num_groups_x")
                            {
                                gpu.gridDim[DIMENSION.X] = right.GetUniqueElement().ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "num_groups_y")
                            {
                                gpu.gridDim[DIMENSION.Y] = right.GetUniqueElement().ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "num_groups_z")
                            {
                                gpu.gridDim[DIMENSION.Z] = right.GetUniqueElement().ConvertToInt32();
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

        private void EvaluateGlobalVariables(IEnumerable<GlobalVariable> declarations)
        {
            foreach (GlobalVariable decl in declarations)
            {
                if (decl.TypedIdent.Type is MapType)
                    Memory.AddGlobalArray(decl.Name);
                if (RegularExpressions.TRACKING_VARIABLE.IsMatch(decl.Name))
                {
                    int index = decl.Name.IndexOf('$');
                    string arrayName = decl.Name.Substring(index);
                    //if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.STANDARD)
                    Memory.AddRaceArrayOffsetVariables(arrayName);
                    MemorySpace space = QKeyValue.FindBoolAttribute(decl.Attributes, "global") ? MemorySpace.GLOBAL : MemorySpace.GROUP_SHARED;
                    Memory.SetMemorySpace(arrayName, space);
                }
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
                else if (constant.Name.Equals("local_id_x$1"))
                    Memory.Store(constant.Name, LocalID1[0]);
                else if (constant.Name.Equals("local_id_y$1"))
                    Memory.Store(constant.Name, LocalID1[1]);
                else if (constant.Name.Equals("local_id_z$1"))
                    Memory.Store(constant.Name, LocalID1[2]);
                else if (constant.Name.Equals("local_id_x$2"))
                    Memory.Store(constant.Name, LocalID2[0]);
                else if (constant.Name.Equals("local_id_y$2"))
                    Memory.Store(constant.Name, LocalID2[1]);
                else if (constant.Name.Equals("local_id_z$2"))
                    Memory.Store(constant.Name, LocalID2[2]);
                else if (constant.Name.Equals("group_id_x$1"))
                    Memory.Store(constant.Name, GlobalID1[0]);
                else if (constant.Name.Equals("group_id_y$1"))
                    Memory.Store(constant.Name, GlobalID1[1]);
                else if (constant.Name.Equals("group_id_z$1"))
                    Memory.Store(constant.Name, GlobalID1[2]);
                else if (constant.Name.Equals("group_id_x$2"))
                    Memory.Store(constant.Name, GlobalID2[0]);
                else if (constant.Name.Equals("group_id_y$2"))
                    Memory.Store(constant.Name, GlobalID2[1]);
                else if (constant.Name.Equals("group_id_z$2"))
                    Memory.Store(constant.Name, GlobalID2[2]);
                else if (constant.Name.Equals("group_id_x"))
                    Memory.Store(constant.Name, GlobalID1[0]);
                else if (constant.Name.Equals("group_id_y"))
                    Memory.Store(constant.Name, GlobalID1[1]);
                else if (constant.Name.Equals("group_id_z"))
                    Memory.Store(constant.Name, GlobalID1[2]);
                
            }
        }

        private void InterpretKernel(Program program, Implementation impl, IEnumerable<Block> headers)
        {
            Print.DebugMessage(String.Format("Interpreting implementation '{0}'", impl.Name), 1);
            try
            {
                // Put formal parameters into a state matching the requires clauses
                foreach (Requires requires in impl.Proc.Requires)
                    EvaluateRequires(requires);
                // Initialise any formal parameters not constrained by requires clauses
                InitialiseFormalParams(impl.InParams);
                // Start intrepreting at the entry basic block
                Block block = impl.Blocks[0];
                // Continue until the exit basic block is reached or we exhaust the loop header count
                while (block != null && GlobalHeaderCount < ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopHeaderLimit)
                {
                    if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopEscapeFactor > 0
                    && headers.Contains(block)
                    && HeaderExecutionCounts.ContainsKey(block)
                    && HeaderExecutionCounts[block] > ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopEscapeFactor)
                    {
                        // If we have exceeded the user-set loop escape factor then go to an exit block
                        block = HeaderToLoopExitBlocks[block][0];
                    }
                    else
                    {
                        if (headers.Contains(block))
                        {
                            GlobalHeaderCount++;
                            if (!HeaderExecutionCounts.ContainsKey(block))
                                HeaderExecutionCounts[block] = 0;
                            HeaderExecutionCounts[block]++;
                        }

                        InterpretBasicBlock(program, impl, block);
                        block = TransferControl(block);
                    }
                }

            }
            catch (UnhandledException e)
            {
                Console.WriteLine(e.ToString());
                Memory.Dump();
            }
        }

        private void EvaluateRequires(Requires requires)
        {
            // The following code currently ignores requires which are implications
            ExprTree tree = new ExprTree(requires.Condition); 
            OpNode<BitVector> root = tree.Root() as OpNode<BitVector>;
            if (root != null)
            {          
                if (root.op == "==" || root.op == "!" || root.op == "&&" || root.op == "!=")
                {
                    foreach (HashSet<Node> nodes in tree)
                    {    
                        foreach (Node node in nodes)
                        {
                            if (node is ScalarSymbolNode<BitVector>)
                            {
                                // Initially assume the boolean variable holds. If it is negated this will be handled
                                // further up in the expression tree
                                ScalarSymbolNode<BitVector> scalar = (ScalarSymbolNode<BitVector>)node;
                                Memory.Store(scalar.symbol, BitVector.True);
                            }
                            else if (node is UnaryNode<BitVector>)
                            {
                                UnaryNode<BitVector> unary = node as UnaryNode<BitVector>;
                                Node child = unary.GetChildren()[0];
                                if (child is ScalarSymbolNode<BitVector>)
                                {
                                    Memory.Store(((ScalarSymbolNode<BitVector>)child).symbol, BitVector.False); 
                                }                         
                            }
                            else if (node is BinaryNode<BitVector>)
                            {
                                BinaryNode<BitVector> binary = node as BinaryNode<BitVector>;
                                if (binary.op == "==")
                                {
                                    LiteralNode<BitVector> right = binary.GetChildren()[1] as LiteralNode<BitVector>;
                                    if (right != null)
                                    {   
                                        ScalarSymbolNode<BitVector> left = binary.GetChildren()[0] as ScalarSymbolNode<BitVector>;    
                                        MapSymbolNode<BitVector> left2 = binary.GetChildren()[0] as MapSymbolNode<BitVector>;
                                        if (left != null)
                                        {
                                            Memory.Store(left.symbol, right.GetUniqueElement());
                                        }
                                        else if (left2 != null)
                                        {
                                            SubscriptExpr subscriptExpr = new SubscriptExpr();
                                            foreach (ExprNode<BitVector> child in left2.GetChildren())
                                            {
                                                BitVector subscript = child.GetUniqueElement();
                                                subscriptExpr.AddIndex(subscript);
                                            }
                                            Memory.Store(left2.basename, subscriptExpr, right.GetUniqueElement());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        private BitVector InitialiseFormalParameter(int width)
        {
            // Boolean types have width 1
            if (width == 1)
            {
                if (Random.Next(0, 2) == 1)
                    return BitVector.True;
                else
                    return BitVector.False;
            }
            else
                return new BitVector(Random.Next(2, 513));
        }

        private void InitialiseFormalParams(List<Variable> formals)
        {
            foreach (Variable v in formals)
            {
                // Only initialise formal parameters not initialised through requires clauses
                // and which can influence control flow
                if (!Memory.Contains(v.Name))
                {
                    //Print.WarningMessage(String.Format("Formal parameter '{0}' not initialised", v.Name));
                    int width;
                    if (v.TypedIdent.Type is BvType)
                    {
                        BvType bv = (BvType)v.TypedIdent.Type;
                        width = bv.Bits;
                    }
                    else if (v.TypedIdent.Type is BasicType)
                    {
                        BasicType basic = (BasicType)v.TypedIdent.Type;
                        if (basic.IsInt)
                            width = 32;
                        else
                            throw new UnhandledException(String.Format("Unhandled basic type '{0}'", basic.ToString()));
                    }
                    else
                        throw new UnhandledException("Unknown data type " + v.TypedIdent.Type.ToString());
                    
                    BitVector initialValue = InitialiseFormalParameter(width);                    
                    Memory.Store(v.Name, initialValue);
                    Print.VerboseMessage(String.Format("Formal parameter '{0}' with type '{1}' is uninitialised. Assigning {2}", 
                        v.Name, v.TypedIdent.Type.ToString(),
                        initialValue.ToString()));
                }
            }
        }

        private void InterpretBasicBlock(Program program, Implementation impl, Block block)
        {
            Print.DebugMessage(String.Format("==========> Entering basic block with label '{0}'", block.Label), 1);
            // Record that this basic block has executed
            Covered.Add(block);
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
                            if (!tree.uninitialised)
                                Memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree.evaluation);
                        }
                        else
                        {
                            SimpleAssignLhs lhs = (SimpleAssignLhs)LhsEval.Item1;
                            ExprTree tree = LhsEval.Item2;
                            if (!tree.uninitialised)
                                Memory.Store(lhs.AssignedVariable.Name, tree.evaluation);
                        }
                    }
                }
                else if (cmd is CallCmd)
                {
                    CallCmd call = cmd as CallCmd;
                    if (RegularExpressions.LOG_READ.IsMatch(call.callee))
                        LogRead(call);
                    else if (RegularExpressions.LOG_WRITE.IsMatch(call.callee))
                        LogWrite(call);
                    else if (RegularExpressions.LOG_ATOMIC.IsMatch(call.callee))
                        LogAtomic(call);
                    else if (RegularExpressions.BUGLE_BARRIER.IsMatch(call.callee))
                        Barrier(call);
                }
                else if (cmd is AssertCmd)
                {
                    AssertCmd assert = cmd as AssertCmd;
                    // Only check asserts which have attributes as these are the conjectured invariants
                    string tag = QKeyValue.FindStringAttribute(assert.Attributes, "tag");
                    if (tag != null)
                    {   
                        ExprTree tree = GetExprTree(assert.Expr);
                        if (!AssertStatus.ContainsKey(assert))
                            AssertStatus[assert] = BitVector.True;
                        if (AssertStatus[assert].Equals(BitVector.True))
                        {
                            EvaluateExprTree(tree);
                            if (!tree.uninitialised && tree.evaluation.Equals(BitVector.False))
                            {
                                Print.VerboseMessage("==========> FALSE " + assert.ToString());
                                AssertStatus[assert] = BitVector.False;
                                MatchCollection matches = RegularExpressions.INVARIANT_VARIABLE.Matches(assert.ToString());
                                string BoogieVariable = null;
                                foreach (Match match in matches)
                                {
                                    foreach (Capture capture in match.Captures)
                                    {
                                        BoogieVariable = capture.Value;
                                    }
                                }
                                Print.ConditionalExitMessage(BoogieVariable != null, "Unable to find Boogie variable");
                                KilledAsserts.Add(BoogieVariable);
                                // Kill the assert in Houdini, which will be invoked after the dynamic analyser
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
                            // Generate a random bit string
                            char[] randomBits = new char[bv.Bits];
                            for (int i = 0; i < bv.Bits; ++i)
                            {
                                if (Random.Next(0, 2) == 1)
                                    randomBits[i] = '1';
                                else
                                    randomBits[i] = '0';
                            }
                            Memory.Store(id.Name, new BitVector(new string(randomBits)));
                        }
                    }
                }
                else if (cmd is AssumeCmd)
                {
                    AssumeCmd assume = cmd as AssumeCmd;
                    ExprTree tree = GetExprTree(assume.Expr);
                    EvaluateExprTree(tree);
                    if (!tree.uninitialised && tree.evaluation.Equals(BitVector.False))
                        Console.WriteLine("ASSUME FALSIFIED: " + assume.Expr.ToString());
                }
                else
                {
                    throw new UnhandledException("Unhandled command: " + cmd.ToString());
                }
            }
        }

        private Block TransferControl(Block block)
        {
            TransferCmd transfer = block.TransferCmd;
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

        private void EvaluateBinaryNode(BinaryNode<BitVector> binary)
        {
            ExprNode<BitVector> left = binary.GetChildren()[0] as ExprNode<BitVector>;
            ExprNode<BitVector> right = binary.GetChildren()[1] as ExprNode<BitVector>;
            
            foreach (BitVector lhs in left.evaluations)
            {
                foreach (BitVector rhs in right.evaluations)
                {
                    if (binary.op.Equals(BinaryOps.IF))
                    {
                        if (lhs.Equals(BitVector.True) && rhs.Equals(BitVector.False))
                            binary.evaluations.Add(BitVector.False);
                        else
                            binary.evaluations.Add(BitVector.True);
                    }
                    else if (RegularExpressions.BVADD.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs + rhs);
                    }
                    else if (RegularExpressions.BVSUB.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs - rhs);
                    }
                    else if (RegularExpressions.BVMUL.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs * rhs);
                    }
                    else if (binary.op.Equals(BinaryOps.EQ))
                    {
                        if (lhs == rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.AND))
                    {
                        if (!(lhs.Equals(BitVector.True) && rhs.Equals(BitVector.True)))
                            binary.evaluations.Add(BitVector.False);
                        else
                            binary.evaluations.Add(BitVector.True);
                    }
                    else if (binary.op.Equals(BinaryOps.OR))
                    {
                        if (lhs.Equals(BitVector.True) || rhs.Equals(BitVector.True))
                            binary.evaluations.Add(BitVector.True);
                    }
                    else if (RegularExpressions.BVAND.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs & rhs);
                    }
                    else if (RegularExpressions.BVOR.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs | rhs);
                    }
                    else if (RegularExpressions.BVSLT.IsMatch(binary.op))
                    {
                        if (lhs < rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVSLE.IsMatch(binary.op))
                    {
                        if (lhs <= rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.IFF))
                    {
                        if ((lhs.Equals(BitVector.True) && rhs.Equals(BitVector.True))
                        || (lhs.Equals(BitVector.False) && rhs.Equals(BitVector.False)))
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.LT))
                    {
                        if (lhs < rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.LTE))
                    {
                        if (lhs <= rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.GT))
                    {
                        if (lhs > rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.GTE))
                    {
                        if (lhs >= rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.NEQ))
                    {
                        if (lhs != rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (binary.op.Equals(BinaryOps.ADD))
                    {
                        binary.evaluations.Add(lhs + rhs);
                    }
                    else if (binary.op.Equals(BinaryOps.SUBTRACT))
                    {
                        binary.evaluations.Add(lhs - rhs);
                    }
                    else if (binary.op.Equals(BinaryOps.MULTIPLY))
                    {
                        binary.evaluations.Add(lhs * rhs);
                    }
                    else if (binary.op.Equals(BinaryOps.DIVIDE))
                    {
                        binary.evaluations.Add(lhs / rhs);
                    }
                    else if (RegularExpressions.BVSGT.IsMatch(binary.op))
                    {
                        if (lhs > rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVSGE.IsMatch(binary.op))
                    {
                        if (lhs >= rhs)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVULT.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                        if (lhsUnsigned < rhsUnsigned)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVULE.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int;
                        if (lhsUnsigned <= rhsUnsigned)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVUGT.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                        if (lhsUnsigned > rhsUnsigned)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVUGE.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                        if (lhsUnsigned >= rhsUnsigned)
                            binary.evaluations.Add(BitVector.True);
                        else
                            binary.evaluations.Add(BitVector.False);
                    }
                    else if (RegularExpressions.BVASHR.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs >> rhs.ConvertToInt32());
                    }
                    else if (RegularExpressions.BVLSHR.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(BitVector.LogicalShiftRight(lhs, rhs.ConvertToInt32()));
                    }
                    else if (RegularExpressions.BVSHL.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs << rhs.ConvertToInt32());
                    }
                    else if (RegularExpressions.BVDIV.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs / rhs);
                    }
                    else if (RegularExpressions.BVXOR.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs ^ rhs);
                    }
                    else if (RegularExpressions.BVSREM.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs % rhs);
                    }
                    else if (RegularExpressions.BVUREM.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                        binary.evaluations.Add(lhsUnsigned % rhsUnsigned);
                    }
                    else if (RegularExpressions.BVSDIV.IsMatch(binary.op))
                    {
                        binary.evaluations.Add(lhs / rhs);
                    }
                    else if (RegularExpressions.BVUDIV.IsMatch(binary.op))
                    {
                        BitVector lhsUnsigned = lhs >= BitVector.Zero ? lhs : lhs & BitVector.Max32Int; 
                        BitVector rhsUnsigned = rhs >= BitVector.Zero ? rhs : rhs & BitVector.Max32Int; 
                        binary.evaluations.Add(lhsUnsigned / rhsUnsigned);
                    }
                    else if (binary.op.Equals("FEQ32") ||
                    binary.op.Equals("FEQ64") ||
                    binary.op.Equals("FGE32") ||
                    binary.op.Equals("FGE64") ||
                    binary.op.Equals("FGT32") ||
                    binary.op.Equals("FGT64") ||
                    binary.op.Equals("FLE32") ||
                    binary.op.Equals("FLE64") ||
                    binary.op.Equals("FLT32") ||
                    binary.op.Equals("FLT64") ||
                    binary.op.Equals("FUNO32") ||
                    binary.op.Equals("FUNO64"))
                    {
                        Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(lhs, rhs, binary.op);
                        if (!FPInterpretations.ContainsKey(FPTriple))
                        {
                            if (Random.Next(0, 2) == 0)
                                FPInterpretations[FPTriple] = BitVector.False;
                            else
                                FPInterpretations[FPTriple] = BitVector.True;
                        }
                        binary.evaluations.Add(FPInterpretations[FPTriple]);
                    }
                    else if (binary.op.Equals("FADD32") ||
                    binary.op.Equals("FADD64") ||
                    binary.op.Equals("FSUB32") ||
                    binary.op.Equals("FSUB64") ||
                    binary.op.Equals("FMUL32") ||
                    binary.op.Equals("FMUL64") ||
                    binary.op.Equals("FDIV32") ||
                    binary.op.Equals("FDIV64") ||
                    binary.op.Equals("FPOW32") ||
                    binary.op.Equals("FPOW64"))
                    {
                        Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(lhs, rhs, binary.op);
                        if (!FPInterpretations.ContainsKey(FPTriple))
                            FPInterpretations[FPTriple] = new BitVector(Random.Next());
                        binary.evaluations.Add(FPInterpretations[FPTriple]);
                    }
                    else
                        throw new UnhandledException("Unhandled bv binary op: " + binary.op);               
                }
            }
        }

        private void EvaluateUnaryNode(UnaryNode<BitVector> unary)
        {
            ExprNode<BitVector> child = (ExprNode<BitVector>)unary.GetChildren()[0];
            if (child.uninitialised)
                unary.uninitialised = true;
            else
            {
                if (unary.op.Equals("!"))
                {
                    if (child.GetUniqueElement().Equals(BitVector.True))
                        unary.evaluations.Add(BitVector.False);
                    else
                        unary.evaluations.Add(BitVector.True);
                }
                else if (unary.op.Equals("FABS32") ||
                         unary.op.Equals("FABS64") ||
                         unary.op.Equals("FCOS32") ||
                         unary.op.Equals("FCOS64") ||
                         unary.op.Equals("FEXP32") ||
                         unary.op.Equals("FEXP64") ||
                         unary.op.Equals("FLOG32") ||
                         unary.op.Equals("FLOG64") ||
                         unary.op.Equals("FPOW32") ||
                         unary.op.Equals("FPOW64") ||
                         unary.op.Equals("FSIN32") ||
                         unary.op.Equals("FSIN64") ||
                         unary.op.Equals("FSQRT32") ||
                         unary.op.Equals("FSQRT64"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(child.GetUniqueElement(), BitVector.Zero, unary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(Random.Next());
                    unary.evaluations.Add(FPInterpretations[FPTriple]);
                }
                else if (RegularExpressions.BVZEXT.IsMatch(unary.op))
                {
                    BitVector ZeroExtended = BitVector.ZeroExtend(child.GetUniqueElement(), 32);
                    unary.evaluations.Add(ZeroExtended);
                }
                else if (RegularExpressions.BVSEXT.IsMatch(unary.op))
                {
                    BitVector SignExtended = BitVector.SignExtend(child.GetUniqueElement(), 32);
                    unary.evaluations.Add(SignExtended);           
                }
                else if (RegularExpressions.CAST_TO_FP.IsMatch(unary.op))
                {
                    BitVector value = child.GetUniqueElement();
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);
                    if (sourceSize == destinationSize)
                        unary.evaluations.Add(value);
                    else
                    {
                        unary.evaluations.Add(BitVector.ZeroExtend(value, destinationSize));
                    }
                }
                else if (RegularExpressions.CAST_TO_INT.IsMatch(unary.op))
                {
                    BitVector value = child.GetUniqueElement();
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);
                    if (sourceSize == destinationSize)
                        unary.evaluations.Add(value);
                    else
                    {
                        unary.evaluations.Add(BitVector.ZeroExtend(value, destinationSize));
                    }
                }
                else if (RegularExpressions.CAST_FP_TO_DOUBLE.IsMatch(unary.op))
                {
                    BitVector ZeroExtended = BitVector.ZeroExtend(child.GetUniqueElement(), 32);
                    unary.evaluations.Add(ZeroExtended);
                }
                else
                    throw new UnhandledException("Unhandled bv unary op: " + unary.op);
            }  
        }

        private void EvaluateExprTree(ExprTree tree)
        {            
            foreach (HashSet<Node> nodes in tree)
            {
                foreach (Node node in nodes)
                {
                    if (node is ScalarSymbolNode<BitVector>)
                    {
                        ScalarSymbolNode<BitVector> _node = node as ScalarSymbolNode<BitVector>;
                        if (RegularExpressions.WATCHDOG_VARIABLE.IsMatch(_node.symbol))
                        {
                            var visitor = new VariablesOccurringInExpressionVisitor();
                            visitor.Visit(tree.expr);
                            string offsetVariable = "";
                            foreach (Variable variable in visitor.GetVariables())
                            {
                                if (RegularExpressions.TRACKING_VARIABLE.IsMatch(variable.Name))
                                {
                                    int index = variable.Name.IndexOf('$');
                                    string arrayName = variable.Name.Substring(index);
                                    string accessType = variable.Name.Substring(0, index);
                                    if (accessType == "_WRITE_HAS_OCCURRED_")
                                        offsetVariable = "_WRITE_OFFSET_" + arrayName;
                                    else if (accessType == "_READ_HAS_OCCURRED_")
                                        offsetVariable = "_READ_OFFSET_" + arrayName;
                                    else
                                        offsetVariable = "_ATOMIC_OFFSET_" + arrayName;
                                    break;
                                }
                            }
                            Print.ConditionalExitMessage(offsetVariable != "", "Unable to find offset variable");
                            foreach (BitVector offset in Memory.GetRaceArrayOffsets(offsetVariable))
                                _node.evaluations.Add(offset);
                        }
                        else if (RegularExpressions.OFFSET_VARIABLE.IsMatch(_node.symbol))
                        {                            
                            foreach (BitVector offset in Memory.GetRaceArrayOffsets(_node.symbol))
                                _node.evaluations.Add(offset);
                        }
                        else
                        {
                            if (!Memory.Contains(_node.symbol))
                                _node.uninitialised = true;
                            else
                                _node.evaluations.Add(Memory.GetValue(_node.symbol));
                        }
                    }
                    else if (node is MapSymbolNode<BitVector>)
                    {
                        MapSymbolNode<BitVector> _node = node as MapSymbolNode<BitVector>;
                        SubscriptExpr subscriptExpr = new SubscriptExpr();
                        foreach (ExprNode<BitVector> child in _node.GetChildren())
                        {
                            if (child.uninitialised)
                                node.uninitialised = true;
                            else
                            {
                                BitVector subscript = child.GetUniqueElement();
                                subscriptExpr.AddIndex(subscript);
                            }
                        }
                        
                        if (!node.uninitialised)
                        {
                            if (!Memory.Contains(_node.basename, subscriptExpr))
                                node.uninitialised = true;
                            else
                                _node.evaluations.Add(Memory.GetValue(_node.basename, subscriptExpr));
                        }
                    }
                    else if (node is BVExtractNode<BitVector>)
                    {
                        BVExtractNode<BitVector> _node = node as BVExtractNode<BitVector>;
                        ExprNode<BitVector> child = (ExprNode<BitVector>)_node.GetChildren()[0];
                        if (child.uninitialised)
                            node.uninitialised = true;
                        else
                        {
                            foreach (BitVector evalChild in child.evaluations)
                            {
                                BitVector eval = BitVector.Slice(evalChild, _node.high, _node.low);
                                _node.evaluations.Add(eval);
                            }
                        }   
                    }
                    else if (node is BVConcatenationNode<BitVector>)
                    {
                        BVConcatenationNode<BitVector> _node = node as BVConcatenationNode<BitVector>;
                        ExprNode<BitVector> one = (ExprNode<BitVector>)_node.GetChildren()[0];
                        ExprNode<BitVector> two = (ExprNode<BitVector>)_node.GetChildren()[1];
                        if (one.uninitialised || two.uninitialised)
                            node.uninitialised = true;
                        else
                        {
                            Print.ConditionalExitMessage(one.evaluations.Count == 1 && two.evaluations.Count == 1, "Unable to process concatentation expression because the children have mutliple evaluations");
                            BitVector eval = BitVector.Concatenate(one.GetUniqueElement(), two.GetUniqueElement());
                            _node.evaluations.Add(eval);
                        }   
                    }
                    else if (node is UnaryNode<BitVector>)
                    {
                        UnaryNode<BitVector> _node = node as UnaryNode<BitVector>;
                        EvaluateUnaryNode(_node);
                        if (_node.evaluations.Count == 0)
                            _node.uninitialised = true;
                    }
                    else if (node is BinaryNode<BitVector>)
                    {
                        BinaryNode<BitVector> _node = (BinaryNode<BitVector>)node;
                        EvaluateBinaryNode(_node);
                        if (_node.evaluations.Count == 0)
                            _node.uninitialised = true;
                    }
                    else if (node is TernaryNode<BitVector>)
                    {
                        TernaryNode<BitVector> _node = node as TernaryNode<BitVector>;
                        ExprNode<BitVector> one = (ExprNode<BitVector>)_node.GetChildren()[0];
                        ExprNode<BitVector> two = (ExprNode<BitVector>)_node.GetChildren()[1];
                        ExprNode<BitVector> three = (ExprNode<BitVector>)_node.GetChildren()[2];
                        if (one.evaluations.Count == 0)
                            node.uninitialised = true;
                        else
                        {
                            if (one.GetUniqueElement().Equals(BitVector.True))
                            {
                                if (two.uninitialised)
                                    node.uninitialised = true;
                                else
                                    _node.evaluations.Add(two.GetUniqueElement());
                            }
                            else
                            {
                                if (three.uninitialised)
                                    node.uninitialised = true;
                                else
                                    _node.evaluations.Add(three.GetUniqueElement());
                            }
                        }
                    }
                } 
            }
            
            ExprNode<BitVector> root = tree.Root() as ExprNode<BitVector>;
            tree.uninitialised = root.uninitialised;
            if (root.evaluations.Count == 1)
                tree.evaluation = root.GetUniqueElement();
            else
            {
                if (root.evaluations.Count == 0)
                    tree.evaluation = BitVector.False;
                else
                {
                    tree.evaluation = BitVector.True;
                    foreach (BitVector eval in root.evaluations)
                    {
                        if (eval.Equals(BitVector.False))
                        {
                            tree.evaluation = BitVector.False;
                            break;
                        }
                    }
                }
            }
        }

        private void Barrier(CallCmd call)
        {
            Print.DebugMessage("In barrier", 10);
            ExprTree groupSharedTree = GetExprTree(call.Ins[0]);
            ExprTree globalTree = GetExprTree(call.Ins[1]);
            EvaluateExprTree(groupSharedTree);
            EvaluateExprTree(globalTree);
            foreach (string name in Memory.GetRaceArrayVariables())
            {
                int index = name.IndexOf('$');
                string arrayName = name.Substring(index);
                if ((Memory.IsInGlobalMemory(arrayName) && globalTree.evaluation.Equals(BitVector.True)) ||
                (Memory.IsInGroupSharedMemory(arrayName) && groupSharedTree.evaluation.Equals(BitVector.True)))
                {
                    if (Memory.GetRaceArrayOffsets(name).Count > 0)
                    {
                        string accessType = name.Substring(0, index);
                        switch (accessType)
                        {
                            case "_WRITE_OFFSET_":
                                {
                                    string accessTracker = "_WRITE_HAS_OCCURRED_" + arrayName; 
                                    Memory.Store(accessTracker, BitVector.False);
                                    break;   
                                }
                            case "_READ_OFFSET_":
                                {
                                    string accessTracker = "_READ_HAS_OCCURRED_" + arrayName; 
                                    Memory.Store(accessTracker, BitVector.False);
                                    break;
                                } 
                            case "_ATOMIC_OFFSET_":
                                {
                                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName; 
                                    Memory.Store(accessTracker, BitVector.False);
                                    break;
                                }
                        }
                    }
                    Memory.ClearRaceArrayOffset(name);
                }
            }
        }

        private void LogRead(CallCmd call)
        {
            Print.DebugMessage("In log read", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            string raceArrayOffsetName = "_READ_OFFSET_" + arrayName;
            if (!Memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_READ_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(Memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find array read offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (!tree1.uninitialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (!tree2.uninitialised)
                {
                    Memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_READ_HAS_OCCURRED_" + arrayName; 
                    Memory.Store(accessTracker, BitVector.True);
                }
            }
        }

        private void LogWrite(CallCmd call)
        {
            Print.DebugMessage("In log write", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            string raceArrayOffsetName = "_WRITE_OFFSET_" + arrayName;
            if (!Memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_WRITE_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(Memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find array write offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (!tree1.uninitialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (!tree2.uninitialised)
                {
                    Memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_WRITE_HAS_OCCURRED_" + arrayName; 
                    Memory.Store(accessTracker, BitVector.True);
                }
            }
        }

        private void LogAtomic(CallCmd call)
        {
            Print.DebugMessage("In log atomic", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            // Evaluate the offset expression 
            Expr offsetExpr = call.Ins[1];
            ExprTree offsetTree = GetExprTree(offsetExpr);
            EvaluateExprTree(offsetTree);
            // Build the subscript expression
            SubscriptExpr subscriptExpr = new SubscriptExpr();
            subscriptExpr.AddIndex(offsetTree.evaluation);
            // For now assume there is only one argument to the atomic function
            Expr argExpr = QKeyValue.FindExprAttribute(call.Attributes, "arg1");
            ExprTree argTree = GetExprTree(argExpr);
            EvaluateExprTree(argTree);
            string atomicFunction = QKeyValue.FindStringAttribute(call.Attributes, "atomic_function");
            switch (atomicFunction)
            {
                case "__atomicAdd_unsigned_int":
                    BitVector currentVal = Memory.GetValue(arrayName, subscriptExpr);
                    BitVector updatedVal = currentVal + argTree.evaluation;
                    Memory.Store(arrayName, subscriptExpr, updatedVal);
                    break;
                default:
                    throw new UnhandledException("Unable to handle atomic function: " + atomicFunction);
            }
        }
    }
}
