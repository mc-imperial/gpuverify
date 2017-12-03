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
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text.RegularExpressions;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;
    using Microsoft.Boogie.Houdini;

    class UnhandledException : Exception
    {
        public UnhandledException(string message)
             : base(message)
        {
        }
    }

    class TimeLimitException : Exception
    {
        public TimeLimitException(string message)
             : base(message)
        {
        }
    }

    internal static class BinaryOps
    {
        public const string OR = "||";
        public const string AND = "&&";
        public const string IF = "==>";
        public const string IFF = "<==>";
        public const string GT = ">";
        public const string GTE = ">=";
        public const string LT = "<";
        public const string LTE = "<=";
        public const string ADD = "+";
        public const string SUBTRACT = "-";
        public const string MULTIPLY = "*";
        public const string DIVIDE = "/";
        public const string NEQ = "!=";
        public const string EQ = "==";
    }

    internal static class RegularExpressions
    {
        public static readonly Regex INVARIANT_VARIABLE = new Regex("_[a-z][0-9]+");

        // Case sensitive
        public static readonly Regex WATCHDOG_VARIABLE = new Regex("_WATCHED_OFFSET", RegexOptions.IgnoreCase);
        public static readonly Regex OFFSET_VARIABLE = new Regex("_(WRITE|READ|ATOMIC)_OFFSET_", RegexOptions.IgnoreCase);
        public static readonly Regex TRACKING_VARIABLE = new Regex("_(WRITE|READ|ATOMIC)_HAS_OCCURRED_", RegexOptions.IgnoreCase);
        public static readonly Regex LOG_READ = new Regex("_LOG_READ_", RegexOptions.IgnoreCase);
        public static readonly Regex LOG_WRITE = new Regex("_LOG_WRITE_", RegexOptions.IgnoreCase);
        public static readonly Regex LOG_ATOMIC = new Regex("_LOG_ATOMIC_", RegexOptions.IgnoreCase);
        public static readonly Regex BUGLE_BARRIER = new Regex("bugle_barrier", RegexOptions.IgnoreCase);
        public static readonly Regex BVSLE = new Regex("BV[0-9]+_SLE", RegexOptions.IgnoreCase);
        public static readonly Regex BVSLT = new Regex("BV[0-9]+_SLT", RegexOptions.IgnoreCase);
        public static readonly Regex BVSGE = new Regex("BV[0-9]+_SGE", RegexOptions.IgnoreCase);
        public static readonly Regex BVSGT = new Regex("BV[0-9]+_SGT", RegexOptions.IgnoreCase);
        public static readonly Regex BVULE = new Regex("BV[0-9]+_ULE", RegexOptions.IgnoreCase);
        public static readonly Regex BVULT = new Regex("BV[0-9]+_ULT", RegexOptions.IgnoreCase);
        public static readonly Regex BVUGE = new Regex("BV[0-9]+_UGE", RegexOptions.IgnoreCase);
        public static readonly Regex BVUGT = new Regex("BV[0-9]+_UGT", RegexOptions.IgnoreCase);
        public static readonly Regex BVASHR = new Regex("BV[0-9]+_ASHR", RegexOptions.IgnoreCase);
        public static readonly Regex BVLSHR = new Regex("BV[0-9]+_LSHR", RegexOptions.IgnoreCase);
        public static readonly Regex BVSHL = new Regex("BV[0-9]+_SHL", RegexOptions.IgnoreCase);
        public static readonly Regex BVADD = new Regex("BV[0-9]+_ADD", RegexOptions.IgnoreCase);
        public static readonly Regex BVSUB = new Regex("BV[0-9]+_SUB", RegexOptions.IgnoreCase);
        public static readonly Regex BVMUL = new Regex("BV[0-9]+_MUL", RegexOptions.IgnoreCase);
        public static readonly Regex BVDIV = new Regex("BV[0-9]+_DIV", RegexOptions.IgnoreCase);
        public static readonly Regex BVAND = new Regex("BV[0-9]+_AND", RegexOptions.IgnoreCase);
        public static readonly Regex BVOR = new Regex("BV[0-9]+_OR", RegexOptions.IgnoreCase);
        public static readonly Regex BVXOR = new Regex("BV[0-9]+_XOR", RegexOptions.IgnoreCase);
        public static readonly Regex BVSREM = new Regex("BV[0-9]+_SREM", RegexOptions.IgnoreCase);
        public static readonly Regex BVUREM = new Regex("BV[0-9]+_UREM", RegexOptions.IgnoreCase);
        public static readonly Regex BVSDIV = new Regex("BV[0-9]+_SDIV", RegexOptions.IgnoreCase);
        public static readonly Regex BVUDIV = new Regex("BV[0-9]+_UDIV", RegexOptions.IgnoreCase);
        public static readonly Regex BVZEXT = new Regex("BV[0-9]+_ZEXT", RegexOptions.IgnoreCase);
        public static readonly Regex BVSEXT = new Regex("BV[0-9]+_SEXT", RegexOptions.IgnoreCase);
        public static readonly Regex CAST_TO_FP = new Regex("(U|S)I[0-9]+_TO_FP[0-9]+", RegexOptions.IgnoreCase);
        public static readonly Regex CAST_TO_INT = new Regex("FP[0-9]+_TO_(U|S)I[0-9]+", RegexOptions.IgnoreCase);
        public static readonly Regex CAST_FP_TO_DOUBLE = new Regex("FP[0-9]+_CONV[0-9]+", RegexOptions.IgnoreCase);
    }

    public class BoogieInterpreter
    {
        // The engine holding all the configuration options
        private DynamicAnalysis Engine;

        // Local and global IDs of the 2 threads modelled in GPUverify
        private BitVector[] LocalID1 = new BitVector[3];
        private BitVector[] LocalID2 = new BitVector[3];
        private BitVector[] GlobalID1 = new BitVector[3];
        private BitVector[] GlobalID2 = new BitVector[3];

        // The GPU configuration
        private GPU gpu = new GPU();

        // The memory for the interpreter
        private Memory memory = new Memory();

        // The expression trees used internally to evaluate Boogie expressions
        private Dictionary<Expr, ExprTree> exprTrees = new Dictionary<Expr, ExprTree>();

        // A basic block label to basic block mapping
        private Dictionary<string, Block> labelToBlock = new Dictionary<string, Block>();

        // The current status of an assert - is it true or false?
        private Dictionary<string, BitVector> assertStatus = new Dictionary<string, BitVector>();

        // Our FP interpretations
        private Dictionary<Tuple<BitVector, BitVector, string>, BitVector> FPInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, BitVector>();

        // Which basic blocks have been covered
        private HashSet<Block> covered = new HashSet<Block>();

        // Keeping track of header execution counts
        private Graph<Block> theLoops;
        private int globalHeaderCount = 0;
        private Dictionary<Block, int> headerExecutionCounts = new Dictionary<Block, int>();

        // The number of times the kernel has been invoked
        private int executions = 0;

        // To randomise group/thread IDs and uninitialised formal parameters
        private Random random;

        // To time execution
        private Stopwatch stopwatch = new Stopwatch();

        public IEnumerable<string> KilledCandidates()
        {
            return assertStatus.Where(x => x.Value.Equals(BitVector.False)).Select(x => x.Key);
        }

        public BoogieInterpreter(DynamicAnalysis engine, Program program)
        {
            this.Engine = engine;

            // If there are no invariants to falsify, return
            if (program.TopLevelDeclarations.OfType<Constant>().Where(item => QKeyValue.FindBoolAttribute(item.Attributes, "existential")).Count() == 0)
                return;

            stopwatch.Start();
            Implementation impl = program.TopLevelDeclarations.OfType<Implementation>().Where(item => QKeyValue.FindBoolAttribute(item.Attributes, "kernel")).First();

            // Seed the random number generator so that it is deterministic
            random = new Random(impl.Name.Length);

            // Build map from label to basic block
            foreach (Block block in impl.Blocks)
                labelToBlock[block.Label] = block;

            this.theLoops = program.ProcessLoops(impl);

            DoInterpretation(program, impl);
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

        private void DoInterpretation(Program program, Implementation impl)
        {
            Print.VerboseMessage("Falsifying invariants with dynamic analysis...");
            try
            {
                do
                {
                    // Reset the memory in readiness for the next execution
                    memory.Clear();
                    foreach (Block header in theLoops.Headers)
                        headerExecutionCounts[header] = 0;

                    EvaulateAxioms(program.TopLevelDeclarations.OfType<Axiom>());
                    EvaluateGlobalVariables(program.TopLevelDeclarations.OfType<GlobalVariable>());
                    Print.DebugMessage(gpu.ToString(), 1);

                    // Set the local thread IDs and group IDs
                    SetLocalIDs();
                    SetGlobalIDs();
                    Print.DebugMessage("Thread 1 local  ID = " + string.Join(", ", new List<BitVector>(LocalID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 1 global ID = " + string.Join(", ", new List<BitVector>(GlobalID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 local  ID = " + string.Join(", ", new List<BitVector>(LocalID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 global ID = " + string.Join(", ", new List<BitVector>(GlobalID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    EvaluateConstants(program.TopLevelDeclarations.OfType<Constant>());
                    InterpretKernel(program, impl);
                    executions++;
                }
                while (globalHeaderCount < Engine.LoopHeaderLimit &&
                       !AllBlocksCovered(impl) &&
                       executions < 5);
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
                if (!covered.Contains(block))
                    return false;
            }

            Print.VerboseMessage("All basic blocks covered");
            return true;
        }

        private void SummarizeKilledInvariants()
        {
            Print.VerboseMessage("Dynamic analysis removed the following candidates:");
            foreach (KeyValuePair<string, BitVector> pair in assertStatus)
            {
                if (pair.Value.Equals(BitVector.False))
                    Print.VerboseMessage(pair.Key);
            }
        }

        private Tuple<BitVector, BitVector> GetID(int dimUpperBound)
        {
            BitVector val1 = new BitVector(random.Next(0, dimUpperBound + 1));
            BitVector val2 = new BitVector(random.Next(0, dimUpperBound + 1));
            return Tuple.Create(val1, val2);
        }

        private void SetLocalIDs()
        {
            Tuple<BitVector, BitVector> dimX = GetID(gpu.blockDim[DIMENSION.X] - 1);
            Tuple<BitVector, BitVector> dimY = GetID(gpu.blockDim[DIMENSION.Y] - 1);
            Tuple<BitVector, BitVector> dimZ = GetID(gpu.blockDim[DIMENSION.Z] - 1);
            LocalID1[0] = dimX.Item1;
            LocalID2[0] = dimX.Item2;
            LocalID1[1] = dimY.Item1;
            LocalID2[1] = dimY.Item2;
            LocalID1[2] = dimZ.Item1;
            LocalID2[2] = dimZ.Item2;
        }

        private void SetGlobalIDs()
        {
            Tuple<BitVector, BitVector> dimX = GetID(gpu.gridDim[DIMENSION.X] - 1);
            Tuple<BitVector, BitVector> dimY = GetID(gpu.gridDim[DIMENSION.Y] - 1);
            Tuple<BitVector, BitVector> dimZ = GetID(gpu.gridDim[DIMENSION.Z] - 1);
            GlobalID1[0] = dimX.Item1;
            GlobalID2[0] = dimX.Item2;
            GlobalID1[1] = dimY.Item1;
            GlobalID2[1] = dimY.Item2;
            GlobalID1[2] = dimZ.Item1;
            GlobalID2[2] = dimZ.Item2;
        }

        private ExprTree GetExprTree(Expr expr)
        {
            if (!exprTrees.ContainsKey(expr))
                exprTrees[expr] = new ExprTree(expr);
            exprTrees[expr].ClearState();
            return exprTrees[expr];
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
                            if (binary.GetChildren()[0] is ScalarSymbolNode && binary.GetChildren()[1] is LiteralNode)
                            {
                                // Assume that equality is actually assignment into the variable of interest
                                search = false;
                                ScalarSymbolNode left = (ScalarSymbolNode)binary.GetChildren()[0];
                                LiteralNode right = (LiteralNode)binary.GetChildren()[1];
                                if (left.symbol == "group_size_x")
                                {
                                    gpu.blockDim[DIMENSION.X] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.X]));
                                }
                                else if (left.symbol == "group_size_y")
                                {
                                    gpu.blockDim[DIMENSION.Y] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Y]));
                                }
                                else if (left.symbol == "group_size_z")
                                {
                                    gpu.blockDim[DIMENSION.Z] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.blockDim[DIMENSION.Z]));
                                }
                                else if (left.symbol == "num_groups_x")
                                {
                                    gpu.gridDim[DIMENSION.X] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.X]));
                                }
                                else if (left.symbol == "num_groups_y")
                                {
                                    gpu.gridDim[DIMENSION.Y] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Y]));
                                }
                                else if (left.symbol == "num_groups_z")
                                {
                                    gpu.gridDim[DIMENSION.Z] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridDim[DIMENSION.Z]));
                                }
                                else if (left.symbol == "global_offset_x")
                                {
                                    gpu.gridOffset[DIMENSION.X] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridOffset[DIMENSION.X]));
                                }
                                else if (left.symbol == "global_offset_y")
                                {
                                    gpu.gridOffset[DIMENSION.Y] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridOffset[DIMENSION.Y]));
                                }
                                else if (left.symbol == "global_offset_z")
                                {
                                    gpu.gridOffset[DIMENSION.Z] = right.evaluation.ConvertToInt32();
                                    memory.Store(left.symbol, new BitVector(gpu.gridOffset[DIMENSION.Z]));
                                }
                                else
                                {
                                    memory.Store(left.symbol, right.evaluation);
                                }
                            }
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
                    memory.AddGlobalArray(decl.Name);
                if (RegularExpressions.TRACKING_VARIABLE.IsMatch(decl.Name))
                {
                    int index = decl.Name.IndexOf('$');
                    string arrayName = decl.Name.Substring(index);
                    memory.AddRaceArrayOffsetVariables(arrayName);
                    MemorySpace space = QKeyValue.FindBoolAttribute(decl.Attributes, "global") ? MemorySpace.GLOBAL : MemorySpace.GROUP_SHARED;
                    memory.SetMemorySpace(arrayName, space);
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
                        memory.Store(constant.Name, BitVector.True);
                    else
                        memory.Store(constant.Name, BitVector.False);
                }
                else if (constant.Name.Equals("local_id_x$1"))
                {
                    memory.Store(constant.Name, LocalID1[0]);
                }
                else if (constant.Name.Equals("local_id_y$1"))
                {
                    memory.Store(constant.Name, LocalID1[1]);
                }
                else if (constant.Name.Equals("local_id_z$1"))
                {
                    memory.Store(constant.Name, LocalID1[2]);
                }
                else if (constant.Name.Equals("local_id_x$2"))
                {
                    memory.Store(constant.Name, LocalID2[0]);
                }
                else if (constant.Name.Equals("local_id_y$2"))
                {
                    memory.Store(constant.Name, LocalID2[1]);
                }
                else if (constant.Name.Equals("local_id_z$2"))
                {
                    memory.Store(constant.Name, LocalID2[2]);
                }
                else if (constant.Name.Equals("group_id_x$1"))
                {
                    memory.Store(constant.Name, GlobalID1[0]);
                }
                else if (constant.Name.Equals("group_id_y$1"))
                {
                    memory.Store(constant.Name, GlobalID1[1]);
                }
                else if (constant.Name.Equals("group_id_z$1"))
                {
                    memory.Store(constant.Name, GlobalID1[2]);
                }
                else if (constant.Name.Equals("group_id_x$2"))
                {
                    memory.Store(constant.Name, GlobalID2[0]);
                }
                else if (constant.Name.Equals("group_id_y$2"))
                {
                    memory.Store(constant.Name, GlobalID2[1]);
                }
                else if (constant.Name.Equals("group_id_z$2"))
                {
                    memory.Store(constant.Name, GlobalID2[2]);
                }
                else if (constant.Name.Equals("group_id_x"))
                {
                    memory.Store(constant.Name, GlobalID1[0]);
                }
                else if (constant.Name.Equals("group_id_y"))
                {
                    memory.Store(constant.Name, GlobalID1[1]);
                }
                else if (constant.Name.Equals("group_id_z"))
                {
                    memory.Store(constant.Name, GlobalID1[2]);
                }
            }
        }

        private void InterpretKernel(Program program, Implementation impl)
        {
            Print.DebugMessage(string.Format("Interpreting implementation '{0}'", impl.Name), 1);
            try
            {
                // Put formal parameters into a state matching the requires clauses
                foreach (Requires requires in impl.Proc.Requires)
                {
                    EvaluateRequires(requires);
                }

                // Initialise any formal parameters not constrained by requires clauses
                InitialiseFormalParams(impl.InParams);

                // Start interpreting at the entry basic block
                Block block = impl.Blocks[0];

                // Continue until the exit basic block is reached or
                // we exhaust the loop header count or
                // we exhaust the time limit
                while (block != null &&
                       globalHeaderCount < Engine.LoopHeaderLimit &&
                       stopwatch.Elapsed.Seconds < Engine.TimeLimit)
                {
                    if (theLoops.Headers.Contains(block))
                    {
                        globalHeaderCount++;
                        headerExecutionCounts[block]++;
                    }

                    InterpretBasicBlock(program, impl, block);
                    block = TransferControl(block);
                }
            }
            catch (UnhandledException e)
            {
                Console.WriteLine(e.ToString());
                memory.Dump();
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
                            if (scalar.type is BvType)
                            {
                                BvType bv = scalar.type as BvType;
                                if (bv.Bits == 1)
                                {
                                    memory.Store(scalar.symbol, BitVector.True);
                                }
                            }
                        }
                        else if (scalar.type is BasicType)
                        {
                            BasicType basic = scalar.type as BasicType;
                            if (basic.IsBool)
                                memory.Store(scalar.symbol, BitVector.True);
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
                                    memory.Store(_child.symbol, BitVector.False);
                            }
                            else if (_child.type is BasicType)
                            {
                                BasicType basic = _child.type as BasicType;
                                if (basic.IsBool)
                                    memory.Store(_child.symbol, BitVector.False);
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
                                    memory.Store(_left.symbol, right.evaluation);
                                }
                                else if (left is MapSymbolNode)
                                {
                                    MapSymbolNode _left = left as MapSymbolNode;
                                    SubscriptExpr subscriptExpr = new SubscriptExpr();
                                    foreach (ExprNode child in _left.GetChildren())
                                        subscriptExpr.indices.Add(child.evaluation);
                                    memory.Store(_left.basename, subscriptExpr, right.evaluation);
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
                if (random.Next(0, 2) == 1)
                    return BitVector.True;
                else
                    return BitVector.False;
            }
            else
            {
                return new BitVector(random.Next(2, 513), width);
            }
        }

        private void InitialiseFormalParams(List<Variable> formals)
        {
            foreach (Variable v in formals)
            {
                // Only initialise formal parameters not initialised through requires clauses
                // and which can influence control flow
                if (!memory.Contains(v.Name))
                {
                    if (v.TypedIdent.Type is BvType)
                    {
                        BvType bv = (BvType)v.TypedIdent.Type;
                        memory.Store(v.Name, InitialiseFormalParameter(bv.Bits));
                    }
                    else if (v.TypedIdent.Type is BasicType)
                    {
                        BasicType basic = (BasicType)v.TypedIdent.Type;
                        if (basic.IsInt)
                            memory.Store(v.Name, InitialiseFormalParameter(32));
                        else
                            throw new UnhandledException(string.Format("Unhandled basic type '{0}'", basic.ToString()));
                    }
                    else if (v.TypedIdent.Type is TypeSynonymAnnotation)
                    {
                        if (v.TypedIdent.Type.ToString() == "functionPtr")
                            memory.Store(v.Name, null);
                        else
                            throw new UnhandledException(string.Format("Unhandled type synonym '{0}'", v.TypedIdent.ToString()));
                    }
                    else
                    {
                        throw new UnhandledException("Unknown data type " + v.TypedIdent.Type.ToString());
                    }
                }
            }
        }

        private void InterpretBasicBlock(Program program, Implementation impl, Block block)
        {
            Print.DebugMessage(string.Format("==========> Entering basic block with label '{0}'", block.Label), 1);

            // Record that this basic block has executed
            covered.Add(block);

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
                    foreach (var lhsEval in assign.Lhss.Zip(evaluations))
                    {
                        if (lhsEval.Item1 is MapAssignLhs)
                        {
                            MapAssignLhs lhs = (MapAssignLhs)lhsEval.Item1;
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
                                ExprTree tree2 = lhsEval.Item2;
                                if (tree2.initialised)
                                    memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree2.evaluation);
                            }
                        }
                        else
                        {
                            SimpleAssignLhs lhs = (SimpleAssignLhs)lhsEval.Item1;
                            ExprTree tree = lhsEval.Item2;
                            if (tree.initialised)
                                memory.Store(lhs.AssignedVariable.Name, tree.evaluation);
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

                        if (assertBoolean != null)
                        {
                            ExprTree tree = GetExprTree(assert.Expr);

                            if (!assertStatus.ContainsKey(assertBoolean))
                                assertStatus[assertBoolean] = BitVector.True;

                            if (assertStatus[assertBoolean].Equals(BitVector.True))
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
                                if (random.Next(0, 2) == 1)
                                    randomBits[i] = '1';
                                else
                                    randomBits[i] = '0';
                            }
                            memory.Store(id.Name, new BitVector(new string(randomBits)));
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
                {
                    throw new UnhandledException("Unhandled command: " + cmd.ToString());
                }
            }
        }

        private void EvaluateAssertWithOffsets(Program program, Implementation impl, ExprTree tree, AssertCmd assert, string assertBoolean)
        {
            // The 'indices' list contains indices into a offset variable set, thus providing a concrete offset value
            // The 'sizes' list is the number of offset values currently being analysed
            List<int> indices = new List<int>();
            List<int> sizes = new List<int>();
            List<Tuple<string, List<BitVector>>> offsetVariableValues = new List<Tuple<string, List<BitVector>>>();
            foreach (string offsetVariable in tree.offsetVariables)
            {
                HashSet<BitVector> offsets = memory.GetRaceArrayOffsets(offsetVariable);
                if (offsets.Count > 0)
                {
                    indices.Add(0);
                    sizes.Add(offsets.Count);
                    offsetVariableValues.Add(Tuple.Create(offsetVariable, offsets.ToList()));
                }
            }

            if (indices.Count > 0)
            {
                int evaluations = 0;
                do
                {
                    // Set up the memory correctly for the selected offset variable
                    for (int i = 0; i < indices.Count; ++i)
                    {
                        Tuple<string, List<BitVector>> offsets = offsetVariableValues[i];
                        BitVector offset = offsets.Item2[indices[i]];
                        if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
                        {
                            memory.Store(offsets.Item1, offset);
                        }
                        else if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE)
                        {
                            memory.Store("_WATCHED_OFFSET_", offset);
                        }
                        else if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE)
                        {
                            int index = offsets.Item1.IndexOf('$');
                            string arrayName = offsets.Item1.Substring(index);
                            memory.Store("_WATCHED_OFFSET_" + arrayName, offset);
                        }
                        else
                        {
                            throw new UnhandledException("Race instrumentation " + RaceInstrumentationUtil.RaceCheckingMethod + " not supported");
                        }
                    }

                    EvaluateAssert(program, impl, tree, assert, assertBoolean);
                    if (assertStatus[assertBoolean] == BitVector.False)
                        break;
                    evaluations++;
                }
                while (CartesianProduct(indices, sizes) && evaluations < 5);
            }
        }

        private void EvaluateAssert(Program program, Implementation impl, ExprTree tree, AssertCmd assert, string assertBoolean)
        {
            EvaluateExprTree(tree);
            if (tree.initialised && tree.evaluation.Equals(BitVector.False))
            {
                Print.VerboseMessage("==========> FALSE " + assert.ToString());
                assertStatus[assertBoolean] = BitVector.False;

                // Tell Houdini about the killed assert
                Houdini.RefutedAnnotation annotation = GPUVerify.GVUtil.GetRefutedAnnotation(program, assertBoolean, impl.Name);
                ConcurrentHoudini.RefutedSharedAnnotations[assertBoolean] = annotation;
            }
        }

        private bool CartesianProduct(List<int> indices, List<int> sizes)
        {
            bool changed = false;
            for (int i = indices.Count - 1; !changed && i > 0; --i)
            {
                if (indices[i] < sizes[i] - 1)
                {
                    indices[i]++;
                    changed = true;
                }
                else
                {
                    indices[i] = 0;
                }
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
                    Block succ = labelToBlock[succLabel];
                    return succ;
                }
                else
                {
                    // Loop through all potential successors and find one whose guard evaluates to true
                    foreach (string succLabel in goto_.labelNames)
                    {
                        Block succ = labelToBlock[succLabel];
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
            {
                return null;
            }

            throw new UnhandledException("Unhandled control transfer command: " + transfer.ToString());
        }

        private bool IsBoolBinaryOp(BinaryNode binary)
        {
            return binary.op.Equals(BinaryOps.IF) ||
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
            RegularExpressions.BVUGE.IsMatch(binary.op);
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
                    if ((left.evaluation.Equals(BitVector.True) && right.evaluation.Equals(BitVector.False)) ||
                        (left.evaluation.Equals(BitVector.False) && right.evaluation.Equals(BitVector.True)))
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
                {
                    throw new UnhandledException("Unhandled bv binary op: " + binary.op);
                }
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
                    binary.evaluation = left.evaluation + right.evaluation;
                }
                else if (RegularExpressions.BVSUB.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation - right.evaluation;
                }
                else if (RegularExpressions.BVMUL.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation * right.evaluation;
                }
                else if (RegularExpressions.BVAND.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation & right.evaluation;
                }
                else if (RegularExpressions.BVOR.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation | right.evaluation;
                }
                else if (binary.op.Equals(BinaryOps.ADD))
                {
                    binary.evaluation = left.evaluation + right.evaluation;
                }
                else if (binary.op.Equals(BinaryOps.SUBTRACT))
                {
                    binary.evaluation = left.evaluation - right.evaluation;
                }
                else if (binary.op.Equals(BinaryOps.MULTIPLY))
                {
                    binary.evaluation = left.evaluation * right.evaluation;
                }
                else if (binary.op.Equals(BinaryOps.DIVIDE))
                {
                    binary.evaluation = left.evaluation / right.evaluation;
                }
                else if (RegularExpressions.BVASHR.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation >> right.evaluation.ConvertToInt32();
                }
                else if (RegularExpressions.BVLSHR.IsMatch(binary.op))
                {
                    binary.evaluation = BitVector.LogicalShiftRight(left.evaluation, right.evaluation.ConvertToInt32());
                }
                else if (RegularExpressions.BVSHL.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation << right.evaluation.ConvertToInt32();
                }
                else if (RegularExpressions.BVDIV.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation / right.evaluation;
                }
                else if (RegularExpressions.BVXOR.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation ^ right.evaluation;
                }
                else if (RegularExpressions.BVSREM.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation % right.evaluation;
                }
                else if (RegularExpressions.BVUREM.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length);
                    binary.evaluation = lhsUnsigned % rhsUnsigned;
                }
                else if (RegularExpressions.BVSDIV.IsMatch(binary.op))
                {
                    binary.evaluation = left.evaluation / right.evaluation;
                }
                else if (RegularExpressions.BVUDIV.IsMatch(binary.op))
                {
                    BitVector lhsUnsigned = left.evaluation >= BitVector.Zero(left.evaluation.Bits.Length) ? left.evaluation : left.evaluation & BitVector.Max(left.evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.evaluation >= BitVector.Zero(right.evaluation.Bits.Length) ? right.evaluation : right.evaluation & BitVector.Max(right.evaluation.Bits.Length);
                    binary.evaluation = lhsUnsigned / rhsUnsigned;
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
                        if (random.Next(0, 2) == 0)
                            FPInterpretations[FPTriple] = BitVector.False;
                        else
                            FPInterpretations[FPTriple] = BitVector.True;
                    }

                    binary.evaluation = FPInterpretations[FPTriple];
                }
                else if (binary.op.Equals("FADD32") ||
                                 binary.op.Equals("FSUB32") ||
                                 binary.op.Equals("FMUL32") ||
                                 binary.op.Equals("FDIV32") ||
                                 binary.op.Equals("FPOW32"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(left.evaluation, right.evaluation, binary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(random.Next(), 32);
                    binary.evaluation = FPInterpretations[FPTriple];
                }
                else if (binary.op.Equals("FADD64") ||
                                 binary.op.Equals("FSUB64") ||
                                 binary.op.Equals("FMUL64") ||
                                 binary.op.Equals("FDIV64") ||
                                 binary.op.Equals("FPOW64"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(left.evaluation, right.evaluation, binary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(random.Next(), 64);
                    binary.evaluation = FPInterpretations[FPTriple];
                }
                else
                {
                    throw new UnhandledException("Unhandled bv binary op: " + binary.op);
                }
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
                                 unary.op.Equals("FCOS32") ||
                                 unary.op.Equals("FEXP32") ||
                                 unary.op.Equals("FFLOOR32") ||
                                 unary.op.Equals("FLOG32") ||
                                 unary.op.Equals("FPOW32") ||
                                 unary.op.Equals("FSIN32") ||
                                 unary.op.Equals("FSQRT32"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(child.evaluation, child.evaluation, unary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(random.Next(), 32);
                    unary.evaluation = FPInterpretations[FPTriple];
                }
                else if (unary.op.Equals("FABS64") ||
                                 unary.op.Equals("FCOS64") ||
                                 unary.op.Equals("FEXP64") ||
                                 unary.op.Equals("FFLOOR64") ||
                                 unary.op.Equals("FLOG64") ||
                                 unary.op.Equals("FPOW64") ||
                                 unary.op.Equals("FSIN64") ||
                                 unary.op.Equals("FSQRT64"))
                {
                    Tuple<BitVector, BitVector, string> FPTriple = Tuple.Create(child.evaluation, child.evaluation, unary.op);
                    if (!FPInterpretations.ContainsKey(FPTriple))
                        FPInterpretations[FPTriple] = new BitVector(random.Next(), 64);
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
                    else if (sourceSize > destinationSize)
                        unary.evaluation = BitVector.Slice(child.evaluation, destinationSize, 0);
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
                    else if (sourceSize > destinationSize)
                        unary.evaluation = BitVector.Slice(child.evaluation, destinationSize, 0);
                    else
                        unary.evaluation = BitVector.ZeroExtend(child.evaluation, destinationSize);
                }
                else if (RegularExpressions.CAST_FP_TO_DOUBLE.IsMatch(unary.op))
                {
                    unary.evaluation = BitVector.ZeroExtend(child.evaluation, 32);
                }
                else if (unary.op.Equals("FUNCPTR_TO_PTR") ||
                         unary.op.Equals("PTR_TO_FUNCPTR"))
                {
                    unary.evaluation = child.evaluation;
                }
                else
                {
                    throw new UnhandledException("Unhandled bv unary op: " + unary.op);
                }
            }
        }

        private void EvaluateExprTree(ExprTree tree)
        {
            foreach (HashSet<Node> nodes in tree)
            {
                foreach (Node node in nodes)
                {
                    if (node is ScalarSymbolNode)
                    {
                        ScalarSymbolNode _node = node as ScalarSymbolNode;
                        if (memory.Contains(_node.symbol))
                            _node.evaluation = memory.GetValue(_node.symbol);
                        else
                            _node.initialised = false;
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
                            if (memory.Contains(_node.basename, subscriptExpr))
                                _node.evaluation = memory.GetValue(_node.basename, subscriptExpr);
                            else
                                _node.initialised = false;
                        }
                    }
                    else if (node is BVExtractNode)
                    {
                        BVExtractNode _node = node as BVExtractNode;
                        ExprNode child = (ExprNode)_node.GetChildren()[0];
                        if (child.initialised)
                            _node.evaluation = BitVector.Slice(child.evaluation, _node.high, _node.low);
                        else
                            _node.initialised = false;
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
                    }
                    else if (node is UnaryNode)
                    {
                        UnaryNode _node = node as UnaryNode;
                        EvaluateUnaryNode(_node);
                    }
                    else if (node is BinaryNode)
                    {
                        BinaryNode _node = node as BinaryNode;
                        if (IsBoolBinaryOp(_node))
                            EvaluateBinaryBoolNode(_node);
                        else
                            EvaluateBinaryNonBoolNode(_node);
                    }
                    else if (node is TernaryNode)
                    {
                        TernaryNode _node = node as TernaryNode;
                        ExprNode one = (ExprNode)_node.GetChildren()[0];
                        ExprNode two = (ExprNode)_node.GetChildren()[1];
                        ExprNode three = (ExprNode)_node.GetChildren()[2];
                        if (!one.initialised)
                        {
                            _node.initialised = false;
                        }
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
            foreach (string name in memory.GetRaceArrayVariables())
            {
                int index = name.IndexOf('$');
                string arrayName = name.Substring(index);
                if ((memory.IsInGlobalMemory(arrayName) && globalTree.evaluation.Equals(BitVector.True)) ||
                            (memory.IsInGroupSharedMemory(arrayName) && groupSharedTree.evaluation.Equals(BitVector.True)))
                {
                    if (memory.GetRaceArrayOffsets(name).Count > 0)
                    {
                        string accessType = name.Substring(0, index);
                        switch (accessType)
                        {
                            case "_WRITE_OFFSET_":
                                {
                                    string accessTracker = "_WRITE_HAS_OCCURRED_" + arrayName;
                                    memory.Store(accessTracker, BitVector.False);
                                    break;
                                }
                            case "_READ_OFFSET_":
                                {
                                    string accessTracker = "_READ_HAS_OCCURRED_" + arrayName;
                                    memory.Store(accessTracker, BitVector.False);
                                    break;
                                }
                            case "_ATOMIC_OFFSET_":
                                {
                                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName;
                                    memory.Store(accessTracker, BitVector.False);
                                    break;
                                }
                        }
                    }

                    memory.ClearRaceArrayOffset(name);
                }
            }
        }

        private void LogRead(CallCmd call)
        {
            Print.DebugMessage("In log read", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            string raceArrayOffsetName = "_READ_OFFSET_" + arrayName;
            if (!memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_READ_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_READ_HAS_OCCURRED_" + arrayName;
                    memory.Store(accessTracker, BitVector.True);
                }
            }
        }

        private void LogWrite(CallCmd call)
        {
            Print.DebugMessage("In log write", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            string raceArrayOffsetName = "_WRITE_OFFSET_" + arrayName;
            if (!memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_WRITE_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_WRITE_HAS_OCCURRED_" + arrayName;
                    memory.Store(accessTracker, BitVector.True);
                }
            }
        }

        private void LogAtomic(CallCmd call)
        {
            Print.DebugMessage("In log atomic", 10);
            int index = call.callee.IndexOf('$');
            string arrayName = call.callee.Substring(index);
            string raceArrayOffsetName = "_ATOMIC_OFFSET_" + arrayName;
            if (!memory.HasRaceArrayVariable(raceArrayOffsetName))
                raceArrayOffsetName = "_ATOMIC_OFFSET_" + arrayName + "$1";
            Print.ConditionalExitMessage(memory.HasRaceArrayVariable(raceArrayOffsetName), "Unable to find offset variable: " + raceArrayOffsetName);
            ExprTree tree1 = GetExprTree(call.Ins[0]);
            EvaluateExprTree(tree1);
            if (tree1.initialised && tree1.evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.evaluation);
                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName;
                    memory.Store(accessTracker, BitVector.True);
                }
            }
        }
    }
}
