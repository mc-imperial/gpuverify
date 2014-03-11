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
    
    internal class DepthFirstSearch
    {
        enum COLOR {WHITE, GREY, BLACK};
        
        private Block start;
        private Graph<Block> cfg;
        private HashSet<Block> allHeaders;
        private HashSet<Block> visitedHeaders = new HashSet<Block>();
        private Dictionary<Block, COLOR> visited = new Dictionary<Block, COLOR>();
        
        public DepthFirstSearch (Block start, Graph<Block> cfg, HashSet<Block> allHeaders)
        {
            this.start = start;
            this.cfg = cfg;
            this.allHeaders = allHeaders;
            foreach (Block block in cfg.Nodes)
            {
                visited[block] = COLOR.WHITE;
            }
            DoDFS(start);
        }
        
        private void DoDFS (Block block)
        {
            visited[block] = COLOR.GREY;
            if (allHeaders.Contains(block) && block != start)
            {
                visitedHeaders.Add(block);
            }
            foreach (Block succ in cfg.Successors(block))
            {
                if (visited[succ] == COLOR.WHITE)
                {
                    DoDFS(succ);   
                }
            }
            visited[block] = COLOR.BLACK;
        }
        
        public HashSet<Block> VisitedHeaders ()
        {
            return visitedHeaders;
        }
    }

    public class BoogieInterpreter
    {
        // Local and global IDs of the 2 threads modelled in GPUverify
        private BitVector[] LocalID1 = new BitVector[3];
        private BitVector[] LocalID2 = new BitVector[3];
        private BitVector[] GlobalID1 = new BitVector[3];
        private BitVector[] GlobalID2 = new BitVector[3];
        
        // The GPU configuration
        private GPU gpu = new GPU();
        
        // The memory for the interpreter
        private Memory Memory = new Memory();
        
        // The expression trees used internally to evaluate Boogie expressions
        private Dictionary<Expr, ExprTree> ExprTrees = new Dictionary<Expr, ExprTree>();
        
        // A basic block label to basic block mapping
        private Dictionary<string, Block> LabelToBlock = new Dictionary<string, Block>();
        
        // The current status of the assert - is it true or false?
        private Dictionary<string, BitVector> AssertStatus = new Dictionary<string, BitVector>();
 
        // Our FP interpretrations
        private Dictionary<Tuple<BitVector, BitVector, string>, BitVector> FPInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, BitVector>();
        
        // Which basic blocks have been covered
        private HashSet<Block> Covered = new HashSet<Block>();
        
        // Keeping trace of header execution counts
        private int GlobalHeaderCount = 0;
        private Dictionary<Block, int> HeaderExecutionCounts = new Dictionary<Block, int>();
        
        // Loop bodies and loop-exit destinations
        private Dictionary<Block, HashSet<Block>> HeaderToLoopBody = new Dictionary<Block, HashSet<Block>>();
        private Dictionary<Block, List<Block>> HeaderToLoopExitBlocks = new Dictionary<Block, List<Block>>();
        
        // Headers whose loops are independent from other loops
        private HashSet<Block> HeadersFromWhichToExitEarly = new HashSet<Block>();
        
        private int Executions = 0;
        private Dictionary<System.Type, System.TimeSpan> NodeToTime = new Dictionary<System.Type, System.TimeSpan>();  
        private Random Random;
        
        public int NumberOfKilledCandidates()
        {
            int numFalseAssigns = 0;
            foreach (KeyValuePair<string, BitVector> pair in AssertStatus)
            {
                if (pair.Value.Equals(BitVector.False))
                    numFalseAssigns++;
            }
            return numFalseAssigns;
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
           
            Graph<Block> loopInfo = program.ProcessLoops(impl);
            // Compute targets of loop exits
            ComputeLoopExits(loopInfo);            
            // Determine whether there are loops that could be executed indepedently
            ComputeDisjointLoops(impl, loopInfo);
            
            DoInterpretation(program, impl, localIDSpecification, globalIDSpecification);
        }
        
        private void ComputeLoopExits(Graph<Block> loopInfo)
        { 
            // Compute loop-exit edges for each natural loop
            foreach (Block header in loopInfo.Headers)
            {
                HeaderToLoopBody[header] = new HashSet<Block>();
                HeaderToLoopExitBlocks[header] = new List<Block>();
                // Build loop body
                foreach (Block tail in loopInfo.BackEdgeNodes(header))
                {
                    HeaderToLoopBody[header].UnionWith(loopInfo.NaturalLoops(header, tail));
                }
                // Now find edges (u, v) where u is in the loop body but v is not
                foreach (Block block in HeaderToLoopBody[header])
                {
                    TransferCmd transfer = block.TransferCmd;
                    if (transfer is GotoCmd)
                    {
                        GotoCmd goto_ = transfer as GotoCmd;
                        if (goto_.labelNames.Count == 1)
                        {
                            string succLabel = goto_.labelNames[0];
                            Block succ = LabelToBlock[succLabel];
                            if (!HeaderToLoopBody[header].Contains(succ))
                                HeaderToLoopExitBlocks[header].Add(succ);
                        }
                        else
                        {
                            foreach (string succLabel in goto_.labelNames)
                            {
                                Block succ = LabelToBlock[succLabel];
                                if (!HeaderToLoopBody[header].Contains(succ))
                                    HeaderToLoopExitBlocks[header].Add(succ);
                            }
                        }
                    }
                }
            }
        }
        
        private void ComputeDisjointLoops(Implementation impl, Graph<Block> loopInfo)
        {
            Dictionary<Block, HashSet<Variable>> HeaderToWriteSet = new Dictionary<Block, HashSet<Variable>>();
            Dictionary<Block, HashSet<Variable>> HeaderToReadSet = new Dictionary<Block, HashSet<Variable>>();
            Dictionary<Block, HashSet<Variable>> HeaderToAssertReadSet = new Dictionary<Block, HashSet<Variable>>();
        
            Graph<Block> cfg = Program.GraphFromImpl(impl);
            foreach (Block header in loopInfo.Headers)
            {
                Tuple<HashSet<Variable>, HashSet<Variable>, HashSet<Variable>> theSets = ComputeWriteAndReadSets(header, HeaderToLoopBody[header]);
                HeaderToWriteSet[header] = theSets.Item1;
                HeaderToReadSet[header] = theSets.Item2;
                HeaderToAssertReadSet[header] = theSets.Item3;
            }
            foreach (Block header in loopInfo.Headers)
            {
                DepthFirstSearch dfs = new DepthFirstSearch(header, cfg, new HashSet<Block>(loopInfo.Headers));
                bool earlyExitPermitted = true;
                foreach (Block reachable in dfs.VisitedHeaders())
                {
                    if (HeaderToWriteSet[header].Intersect(HeaderToAssertReadSet[reachable]).Any())
                        earlyExitPermitted = false;
                }
                if (earlyExitPermitted)
                {
                    Print.VerboseMessage("Early exit permitted from " + header);
                    HeadersFromWhichToExitEarly.Add(header);
                }
            }
        }

        private Tuple<HashSet<Variable>, HashSet<Variable>, HashSet<Variable>> ComputeWriteAndReadSets(Block header, HashSet<Block> loopBody)
        {
            HashSet<Variable> writeSet = new HashSet<Variable>();
            HashSet<Variable> readSet = new HashSet<Variable>();
            HashSet<Variable> assertReadSet = new HashSet<Variable>();
            var readVisitor = new VariablesOccurringInExpressionVisitor();
            var assertReadVisitor = new VariablesOccurringInExpressionVisitor();
            foreach (Block block in loopBody)
            {
                foreach (AssignCmd assignment in block.Cmds.OfType<AssignCmd>())
                {
                    List<Variable> written = new List<Variable>();
                    assignment.AddAssignedVariables(written);
                    foreach (Variable variable in written)
                    {
                        writeSet.Add(variable);
                    }
                    foreach (Expr rhs in assignment.Rhss)
                    {
                        readVisitor.Visit(rhs);
                    }
                    foreach (Variable variable in readVisitor.GetVariables())
                    {
                        readSet.Add(variable);
                    }
                }
                foreach (AssertCmd assert in header.Cmds.OfType<AssertCmd>())
                {
                    assertReadVisitor.Visit(assert);
                    foreach (Variable variable in assertReadVisitor.GetVariables())
                        assertReadSet.Add(variable);
                }
            }
            return Tuple.Create(writeSet, readSet, assertReadSet);
        }
        
        private void DoInterpretation (Program program, Implementation impl, Tuple<int, int, int> localIDSpecification, Tuple<int, int, int> globalIDSpecification)
        {
            Print.VerboseMessage("Falsyifying invariants with dynamic analysis...");
            try
            {  
                do
                {
//                    nodeToTime[typeof(UnaryNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(BinaryNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(TernaryNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(BVExtractNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(BVConcatenationNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(MapSymbolNode)] = System.TimeSpan.Zero;
//                    nodeToTime[typeof(ScalarSymbolNode)] = System.TimeSpan.Zero;
                
                    // Reset the memory in readiness for the next execution
                    Memory.Clear();
                    foreach (Block header in HeaderToLoopBody.Keys)
                    {
                        HeaderExecutionCounts[header] = 0;   
                    }
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
                    InterpretKernel(program, impl);
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
            Print.VerboseMessage("Dynamic analysis removed the following candidates:");
            foreach (KeyValuePair<string, BitVector> pair in AssertStatus)
            {
                if (pair.Value.Equals(BitVector.False))
                    Print.VerboseMessage(pair.Key);
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
                    if (node is BinaryNode)
                    {
                        BinaryNode binary = (BinaryNode)node;
                        if (binary.op == "==")
                        {
                            // Assume that equality is actually assignment into the variable of interest
                            search = false;
                            ScalarSymbolNode left = (ScalarSymbolNode)binary.GetChildren()[0];
                            LiteralNode right = (LiteralNode)binary.GetChildren()[1];
                            if (left.symbol == "group_size_x")
                            {
                                gpu.blockDim[DIMENSION.X] = right.evaluation.ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "group_size_y")
                            {
                                gpu.blockDim[DIMENSION.Y] = right.evaluation.ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "group_size_z")
                            {
                                gpu.blockDim[DIMENSION.Z] = right.evaluation.ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Z]));
                            }
                            else if (left.symbol == "num_groups_x")
                            {
                                gpu.gridDim[DIMENSION.X] = right.evaluation.ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.X]));
                            }
                            else if (left.symbol == "num_groups_y")
                            {
                                gpu.gridDim[DIMENSION.Y] = right.evaluation.ConvertToInt32();
                                Memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Y]));
                            }
                            else if (left.symbol == "num_groups_z")
                            {
                                gpu.gridDim[DIMENSION.Z] = right.evaluation.ConvertToInt32();
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

        private void InterpretKernel(Program program, Implementation impl)
        {
            Print.DebugMessage(String.Format("Interpreting implementation '{0}'", impl.Name), 1);
            try
            {
                // Put formal parameters into a state matching the requires clauses
                foreach (Requires requires in impl.Proc.Requires)
                {
                    EvaluateRequires(requires);
                }
                // Initialise any formal parameters not constrained by requires clauses
                InitialiseFormalParams(impl.InParams);
                // Start intrepreting at the entry basic block
                Block block = impl.Blocks[0];
                // Continue until the exit basic block is reached or we exhaust the loop header count
                while (block != null 
                    && GlobalHeaderCount < ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopHeaderLimit)
                {
                    if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopEscapeFactor > 0
                        && HeaderToLoopBody.Keys.Contains(block)
                        && HeaderExecutionCounts.ContainsKey(block)
                        && HeaderExecutionCounts[block] > ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisLoopEscapeFactor)
                    {
                        // If we have exceeded the user-set loop escape factor then go to an exit block
                        block = HeaderToLoopExitBlocks[block][0];
                    }
                    else if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicAnalysisSoundLoopEscaping 
                        && HeaderToLoopBody.Keys.Contains(block)
                        && HeadersFromWhichToExitEarly.Contains(block)
                        && HeaderExecutionCounts[block] > 1)
                    {
                        block = HeaderToLoopExitBlocks[block][0];
                    }
                    else
                    {
                        if (HeaderToLoopBody.Keys.Contains(block))
                        {
                            GlobalHeaderCount++;
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
            EvaluateExprTree(tree);
            foreach (HashSet<Node> nodes in tree)
            {
                foreach (Node node in nodes)
                {
                    if (node is ScalarSymbolNode)
                    {
                        ScalarSymbolNode scalar = node as ScalarSymbolNode;
                        if (scalar.type.IsBv)
                        {
                            BvType bv = scalar.type as BvType;
                            if (bv.Bits == 1)
                                Memory.Store(scalar.symbol, BitVector.True);
                        }
                        else if (scalar.type is BasicType)
                        {
                            BasicType basic = scalar.type as BasicType;
                            if (basic.IsBool)
                                Memory.Store(scalar.symbol, BitVector.True);
                        }
                    }
                    else if (node is UnaryNode)
                    {
                        UnaryNode unary = node as UnaryNode;
                        ExprNode child = unary.GetChildren()[0] as ExprNode; 
                        if (unary.op == "!" && child is ScalarSymbolNode)
                        {
                            ScalarSymbolNode _child = child as ScalarSymbolNode;
                            if (_child.type.IsBv)
                            {
                                BvType bv = _child.type as BvType;
                                if (bv.Bits == 1)
                                    Memory.Store(_child.symbol, BitVector.False);
                            }
                            else if (_child.type is BasicType)
                            {
                                BasicType basic = _child.type as BasicType;
                                if (basic.IsBool)
                                    Memory.Store(_child.symbol, BitVector.False);
                            }
                        }
                    }
                    else if (node is BinaryNode)
                    {
                        BinaryNode binary = node as BinaryNode;
                        if (binary.op == "==")
                        {
                            ExprNode left = binary.GetChildren()[0] as ExprNode; 
                            ExprNode right = binary.GetChildren()[1] as ExprNode; 
                            if (right.initialised)
                            {  
                                if (left is ScalarSymbolNode)
                                {
                                    ScalarSymbolNode _left = left as ScalarSymbolNode;
                                    Memory.Store(_left.symbol, right.evaluation);
                                }
                                else if (left is MapSymbolNode)
                                {
                                    MapSymbolNode _left = left as MapSymbolNode;
                                    SubscriptExpr subscriptExpr = new SubscriptExpr();
                                    foreach (ExprNode child in _left.GetChildren())
                                        subscriptExpr.indices.Add(child.evaluation);
                                    Memory.Store(_left.basename, subscriptExpr, right.evaluation);
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
                return new BitVector(Random.Next(2, 513), width);
        }

        private void InitialiseFormalParams(List<Variable> formals)
        {
            foreach (Variable v in formals)
            {
                // Only initialise formal parameters not initialised through requires clauses
                // and which can influence control flow
                if (!Memory.Contains(v.Name))
                {
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
                        v.Name, v.TypedIdent.Type.ToString(), initialValue.ToString()));
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
                //Console.WriteLine(cmd);
                //Memory.Dump();
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
                                ExprTree tree = GetExprTree(index);
                                EvaluateExprTree(tree);
                                if (tree.initialised)
                                    subscriptExpr.indices.Add(tree.evaluation);
                            }
                            if (subscriptExpr.indices.Count > 0)
                            {
                                ExprTree tree2 = LhsEval.Item2;
                                if (tree2.initialised)
                                    Memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree2.evaluation);
                            }
                        }
                        else
                        {
                            SimpleAssignLhs lhs = (SimpleAssignLhs)LhsEval.Item1;
                            ExprTree tree = LhsEval.Item2;
                            if (tree.initialised)
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
                    // Only check asserts which have attributes as these are the candidate invariants
                    string tag = QKeyValue.FindStringAttribute(assert.Attributes, "tag");
                    if (tag != null)
                    {
                        MatchCollection matches = RegularExpressions.INVARIANT_VARIABLE.Matches(assert.ToString());
                        string assertBoolean = null;
                        foreach (Match match in matches)
                        {
                            foreach (Capture capture in match.Captures)
                            {
                                assertBoolean = capture.Value;
                            }
                        }
                        ExprTree tree = GetExprTree(assert.Expr);
                        if (!AssertStatus.ContainsKey(assertBoolean))
                            AssertStatus[assertBoolean] = BitVector.True;
                        if (AssertStatus[assertBoolean].Equals(BitVector.True))
                        {
                            // Does the expression tree have offset variables?
                            if (tree.offsetVariables.Count > 0)
                            {
                                // If so, evaluate the expression tree using the Cartesian product of all
                                // distinct offset values 
                                EvaluateAssertWithOffsets(program, impl, tree, assert, assertBoolean);
                            }
                            else
                            {
                                // If not, it's a straightforward evaluation
                                EvaluateAssert(program, impl, tree, assert, assertBoolean);
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
                    if (tree.initialised && tree.evaluation.Equals(BitVector.False))
                        Console.WriteLine("ASSUME FALSIFIED: " + assume.Expr.ToString());
                }
                else
                    throw new UnhandledException("Unhandled command: " + cmd.ToString());
            }
        }
        
        private void EvaluateAssertWithOffsets (Program program, Implementation impl, ExprTree tree, AssertCmd assert, string assertBoolean)
        {
            // The 'indices' list contains indices into a offset variable set, thus providing a concrete offset value 
            // The 'sizes' list is the number of offset values currently being analysed
            List<int> indices = new List<int>();
            List<int> sizes = new List<int>();
            List<Tuple<string, List<BitVector>>> offsetVariableValues = new List<Tuple<string, List<BitVector>>>();
            foreach (string offsetVariable in tree.offsetVariables)
            {
                HashSet<BitVector> offsets = Memory.GetRaceArrayOffsets(offsetVariable);
                if (offsets.Count > 0)
                {
                    indices.Add(0);
                    sizes.Add(offsets.Count);
                    offsetVariableValues.Add(Tuple.Create(offsetVariable, offsets.ToList()));
                }
            }
            if (indices.Count > 0)
            {
                int exprEvaluations = 0;
                do
                {
                    // Set up the memory correctly for the selected offset variable
                    for (int i = 0; i < indices.Count; ++i)
                    {
                        Tuple<string, List<BitVector>> offsets = offsetVariableValues[i];
                        BitVector offset = offsets.Item2[indices[i]];
                        if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.STANDARD)                
                            Memory.Store(offsets.Item1, offset);
                        else if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE)
                            Memory.Store("_WATCHED_OFFSET_", offset);
                        else if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE)
                        {
                            int index = offsets.Item1.IndexOf('$');
                            string arrayName = offsets.Item1.Substring(index);
                            Memory.Store("_WATCHED_OFFSET_" + arrayName, offset);
                        }
                        else
                            throw new UnhandledException("Race instrumentation " + RaceInstrumentationUtil.RaceCheckingMethod + " not supported");
                    }
                    exprEvaluations++;
                    EvaluateAssert(program, impl, tree, assert, assertBoolean);
                    if (AssertStatus[assertBoolean] == BitVector.False)
                        break;
                }
                while (CartesianProduct(indices, sizes) && exprEvaluations < 5);
            }
        }

        private void EvaluateAssert(Program program, Implementation impl, ExprTree tree, AssertCmd assert, string assertBoolean)
        {
            EvaluateExprTree(tree);
            if (tree.initialised && tree.evaluation.Equals(BitVector.False))
            {
                Print.VerboseMessage("==========> FALSE " + assert.ToString());
                AssertStatus[assertBoolean] = BitVector.False;
                // Tell Houdini about the killed assert
                ConcurrentHoudini.RefutedAnnotation annotation = GPUVerify.GVUtil.getRefutedAnnotation(program, assertBoolean, impl.Name);
                ConcurrentHoudini.RefutedSharedAnnotations[assertBoolean] = annotation;
            }
        }

        private bool CartesianProduct(List<int> indices, List<int> sizes)
        {
            bool changed = false;
            bool finished = false;
            for (int i = indices.Count - 1; !changed && !finished; --i)
            {
                if (indices[i] < sizes[i] - 1)
                {
                    indices[i]++;
                    changed = true;
                }
                else
                    indices[i] = 0;
                finished = i == 0;
            }
            return changed;
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

        private bool IsBoolBinaryOp(BinaryNode binary)
        {
            return (binary.op.Equals(BinaryOps.IF) ||
            binary.op.Equals(BinaryOps.IFF) ||
            binary.op.Equals(BinaryOps.AND) ||
            binary.op.Equals(BinaryOps.OR) ||
            binary.op.Equals(BinaryOps.NEQ) ||
            binary.op.Equals(BinaryOps.EQ) ||
            binary.op.Equals(BinaryOps.LT) ||
            binary.op.Equals(BinaryOps.LTE) ||
            binary.op.Equals(BinaryOps.GT) ||
            binary.op.Equals(BinaryOps.GTE) ||
            RegularExpressions.BVSLT.IsMatch(binary.op) ||
            RegularExpressions.BVSLE.IsMatch(binary.op) ||
            RegularExpressions.BVSGT.IsMatch(binary.op) ||
            RegularExpressions.BVSGE.IsMatch(binary.op) ||
            RegularExpressions.BVULT.IsMatch(binary.op) ||
            RegularExpressions.BVULE.IsMatch(binary.op) ||
            RegularExpressions.BVUGT.IsMatch(binary.op) ||
            RegularExpressions.BVUGE.IsMatch(binary.op));
        }

        private void EvaluateBinaryBoolNode(BinaryNode binary)
        {
            ExprNode left = binary.GetChildren()[0] as ExprNode;
            ExprNode right = binary.GetChildren()[1] as ExprNode;
   
            binary.initialised = left.initialised && right.initialised;
            if (binary.initialised)
            { 
                if (binary.op.Equals(BinaryOps.IF))
                {
                    if (left.evaluation.Equals(BitVector.True) && right.evaluation.Equals(BitVector.False))
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;     
                }
                else if (binary.op.Equals(BinaryOps.EQ))
                {
                    if (left.evaluation != right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.AND))
                {
                    if (!(left.evaluation.Equals(BitVector.True) && right.evaluation.Equals(BitVector.True)))
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.OR))
                {
                    if (left.evaluation.Equals(BitVector.True) || right.evaluation.Equals(BitVector.True))
                        binary.evaluation = BitVector.True;
                    else
                        binary.evaluation = BitVector.False;
                }
                else if (RegularExpressions.BVSLT.IsMatch(binary.op))
                {
                    if (left.evaluation >= right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVSLE.IsMatch(binary.op))
                {
                    if (left.evaluation > right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.IFF))
                {
                    if ((left.evaluation.Equals(BitVector.True) && right.evaluation.Equals(BitVector.False)) || (left.evaluation.Equals(BitVector.False) && right.evaluation.Equals(BitVector.True)))
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.LT))
                {
                    if (left.evaluation >= right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.LTE))
                {
                    if (left.evaluation > right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.GT))
                {
                    if (left.evaluation <= right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.GTE))
                {
                    if (left.evaluation < right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (binary.op.Equals(BinaryOps.NEQ))
                {
                    if (left.evaluation == right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVSGT.IsMatch(binary.op))
                {
                    if (left.evaluation <= right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVSGE.IsMatch(binary.op))
                {
                    if (left.evaluation < right.evaluation)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVULT.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length); 
                    if (lhsUnsigned >= rhsUnsigned)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVULE.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length);
                    if (lhsUnsigned > rhsUnsigned)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVUGT.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length); 
                    if (lhsUnsigned <= rhsUnsigned)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else if (RegularExpressions.BVUGE.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length); 
                    if (lhsUnsigned < rhsUnsigned)
                        binary.evaluation = BitVector.False;
                    else
                        binary.evaluation = BitVector.True;
                }
                else
                    throw new UnhandledException("Unhandled bv binary op: " + binary.op);    
            }         
        }

        private void EvaluateBinaryNonBoolNode(BinaryNode binary)
        {
            ExprNode left = binary.GetChildren()[0] as ExprNode;
            ExprNode right = binary.GetChildren()[1] as ExprNode;
            
            binary.initialised = left.initialised && right.initialised;
            if (binary.initialised)
            {                  
                if (RegularExpressions.BVADD.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation + right.evaluation);
                }
                else if (RegularExpressions.BVSUB.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation - right.evaluation);
                }
                else if (RegularExpressions.BVMUL.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation * right.evaluation);
                }
                else if (RegularExpressions.BVAND.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation & right.evaluation);
                }
                else if (RegularExpressions.BVOR.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation | right.evaluation);
                }
                else if (binary.op.Equals(BinaryOps.ADD))
                {
                    binary.evaluation = (left.evaluation + right.evaluation);
                }
                else if (binary.op.Equals(BinaryOps.SUBTRACT))
                {
                    binary.evaluation = (left.evaluation - right.evaluation);
                }
                else if (binary.op.Equals(BinaryOps.MULTIPLY))
                {
                    binary.evaluation = (left.evaluation * right.evaluation);
                }
                else if (binary.op.Equals(BinaryOps.DIVIDE))
                {
                    binary.evaluation = (left.evaluation / right.evaluation);
                }
                else if (RegularExpressions.BVASHR.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation >> right.evaluation.ConvertToInt32());
                }
                else if (RegularExpressions.BVLSHR.IsMatch(binary.op))
                {
                    binary.evaluation = (BitVector.LogicalShiftRight(left.evaluation, right.evaluation.ConvertToInt32()));
                }
                else if (RegularExpressions.BVSHL.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation << right.evaluation.ConvertToInt32());
                }
                else if (RegularExpressions.BVDIV.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation / right.evaluation);
                }
                else if (RegularExpressions.BVXOR.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation ^ right.evaluation);
                }
                else if (RegularExpressions.BVSREM.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation % right.evaluation);
                }
                else if (RegularExpressions.BVUREM.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length); 
                    binary.evaluation = (lhsUnsigned % rhsUnsigned);
                }
                else if (RegularExpressions.BVSDIV.IsMatch(binary.op))
                {
                    binary.evaluation = (left.evaluation / right.evaluation);
                }
                else if (RegularExpressions.BVUDIV.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length); 
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length); 
                    binary.evaluation = (lhsUnsigned / rhsUnsigned);
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
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(left.evaluation, right.evaluation, binary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                    {
                        if (Random.Next(0, 2) == 0)
                            FPInterpretations[FPTriple] = BitVector.False;
                        else
                            FPInterpretations[FPTriple] = BitVector.True;
                    }
                    binary.evaluation = (FPInterpretations[FPTriple]);
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
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(left.evaluation, right.evaluation, binary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(Random.Next());
                    binary.evaluation = (FPInterpretations[FPTriple]);
                }
                else
                    throw new UnhandledException("Unhandled bv binary op: " + binary.op);
            }              
        }

        private void EvaluateUnaryNode(UnaryNode unary)
        {
            ExprNode child = unary.GetChildren()[0] as ExprNode;        
            unary.initialised = child.initialised;
            if (unary.initialised)
            {
                if (unary.op.Equals("!"))
                {
                    if (child.evaluation.Equals(BitVector.True))
                        unary.evaluation = BitVector.False;
                    else
                        unary.evaluation = BitVector.True;
                }
                else if (unary.op.Equals("FABS32") ||
                         unary.op.Equals("FABS64") ||
                         unary.op.Equals("FCOS32") ||
                         unary.op.Equals("FCOS64") ||
                         unary.op.Equals("FEXP32") ||
                         unary.op.Equals("FEXP64") ||
                         unary.op.Equals("FFLOOR32") ||
                         unary.op.Equals("FFLOOR64") ||
                         unary.op.Equals("FLOG32") ||
                         unary.op.Equals("FLOG64") ||
                         unary.op.Equals("FPOW32") ||
                         unary.op.Equals("FPOW64") ||
                         unary.op.Equals("FSIN32") ||
                         unary.op.Equals("FSIN64") ||
                         unary.op.Equals("FSQRT32") ||
                         unary.op.Equals("FSQRT64"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(child.evaluation, child.evaluation, unary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(Random.Next());
                    unary.evaluation = FPInterpretations[FPTriple];
                }
                else if (RegularExpressions.BVZEXT.IsMatch(unary.op))
                {
                    int width = 32;
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    if (matches.Count == 2)
                        width = Convert.ToInt32(matches[1].Value);
                    unary.evaluation = BitVector.ZeroExtend(child.evaluation, width);
                }
                else if (RegularExpressions.BVSEXT.IsMatch(unary.op))
                {
                    int width = 32;
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    if (matches.Count == 2)
                        width = Convert.ToInt32(matches[1].Value);
                    unary.evaluation = BitVector.SignExtend(child.evaluation, width);        
                }
                else if (RegularExpressions.CAST_TO_FP.IsMatch(unary.op))
                {
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);
                    if (sourceSize == destinationSize)
                        unary.evaluation = child.evaluation;
                    else
                        unary.evaluation = BitVector.ZeroExtend(child.evaluation, destinationSize);
                }
                else if (RegularExpressions.CAST_TO_INT.IsMatch(unary.op))
                {
                    MatchCollection matches = Regex.Matches(unary.op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);
                    if (sourceSize == destinationSize)
                        unary.evaluation = child.evaluation;
                    else
                        unary.evaluation = BitVector.ZeroExtend(child.evaluation, destinationSize);
                }
                else if (RegularExpressions.CAST_FP_TO_DOUBLE.IsMatch(unary.op))
                {
                    unary.evaluation = BitVector.ZeroExtend(child.evaluation, 32);
                }
                else
                    throw new UnhandledException("Unhandled bv unary op: " + unary.op);
            }  
        }

        private void EvaluateExprTree(ExprTree tree)
        {            
            Stopwatch timer = new Stopwatch();
            foreach (HashSet<Node> nodes in tree)
            {
                foreach (Node node in nodes)
                {
                    if (node is ScalarSymbolNode)
                    {
                        ScalarSymbolNode _node = node as ScalarSymbolNode;
                        if (Memory.Contains(_node.symbol))
                            _node.evaluation = Memory.GetValue(_node.symbol);
                        else
                            _node.initialised = false;
                        //nodeToTime[typeof(ScalarSymbolNode)] += timer.Elapsed;
                    }
                    else if (node is MapSymbolNode)
                    {
                        MapSymbolNode _node = node as MapSymbolNode;
                        SubscriptExpr subscriptExpr = new SubscriptExpr();
                        foreach (ExprNode child in _node.GetChildren())
                        {
                            if (child.initialised)
                                subscriptExpr.indices.Add(child.evaluation);
                            else
                                _node.initialised = false;
                        }
                        
                        if (node.initialised)
                        {
                            if (Memory.Contains(_node.basename, subscriptExpr))
                                _node.evaluation = Memory.GetValue(_node.basename, subscriptExpr);
                            else
                                _node.initialised = false;
                        }
                        //nodeToTime[typeof(MapSymbolNode)] += timer.Elapsed;
                    }
                    else if (node is BVExtractNode)
                    {
                        BVExtractNode _node = node as BVExtractNode;
                        ExprNode child = (ExprNode)_node.GetChildren()[0];
                        if (child.initialised)
                            _node.evaluation = BitVector.Slice(child.evaluation, _node.high, _node.low);
                        else
                            _node.initialised = false;
                        //nodeToTime[typeof(BVExtractNode)] += timer.Elapsed;
                    }
                    else if (node is BVConcatenationNode)
                    {
                        BVConcatenationNode _node = node as BVConcatenationNode;
                        ExprNode one = (ExprNode)_node.GetChildren()[0];
                        ExprNode two = (ExprNode)_node.GetChildren()[1];
                        if (one.initialised && two.initialised)
                            _node.evaluation = BitVector.Concatenate(one.evaluation, two.evaluation);
                        else
                            _node.initialised = false;
                        //nodeToTime[typeof(BVConcatenationNode)] += timer.Elapsed;
                    }
                    else if (node is UnaryNode)
                    {
                        UnaryNode _node = node as UnaryNode;
                        timer.Start();  
                        EvaluateUnaryNode(_node);
                        timer.Stop();
                        //nodeToTime[typeof(UnaryNode)] += timer.Elapsed;
                    }
                    else if (node is BinaryNode)
                    {
                        BinaryNode _node = node as BinaryNode;
                        if (IsBoolBinaryOp(_node))
                            EvaluateBinaryBoolNode(_node);
                        else
                            EvaluateBinaryNonBoolNode(_node);
                        //nodeToTime[typeof(BinaryNode)] += timer.Elapsed;
                    }
                    else if (node is TernaryNode)
                    {
                        TernaryNode _node = node as TernaryNode;
                        ExprNode one = (ExprNode)_node.GetChildren()[0];
                        ExprNode two = (ExprNode)_node.GetChildren()[1];
                        ExprNode three = (ExprNode)_node.GetChildren()[2];
                        if (!one.initialised)
                            _node.initialised = false;
                        else
                        {
                            if (one.evaluation.Equals(BitVector.True))
                            {
                                if (two.initialised)
                                    _node.evaluation = two.evaluation;
                                else
                                    _node.initialised = false;
                            }
                            else
                            {
                                if (three.initialised)
                                    _node.evaluation = three.evaluation;
                                else
                                    _node.initialised = false;
                            }
                        }
                        //nodeToTime[typeof(TernaryNode)] += timer.Elapsed;
                    }
                } 
            }
            
            ExprNode root = tree.Root() as ExprNode;
            tree.initialised = root.initialised;
            tree.evaluation = root.evaluation;
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
            Print.ConditionalExitMessage(Memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
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
            Print.ConditionalExitMessage(Memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
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
            string raceArrayOffsetName = "_ATOMIC_OFFSET_" + arrayName;
            if (!Memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_ATOMIC_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(Memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
                {
                    Memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName; 
                    Memory.Store(accessTracker, BitVector.True);
                }
            }
        }
    }
}
