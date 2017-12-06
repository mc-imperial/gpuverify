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

    public class BoogieInterpreter
    {
        // The engine holding all the configuration options
        private DynamicAnalysis engine;

        // Local and global IDs of the 2 threads modelled in GPUverify
        private BitVector[] localID1 = new BitVector[3];
        private BitVector[] localID2 = new BitVector[3];
        private BitVector[] globalID1 = new BitVector[3];
        private BitVector[] globalID2 = new BitVector[3];

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
        private Dictionary<Tuple<BitVector, BitVector, string>, BitVector> fpInterpretations = new Dictionary<Tuple<BitVector, BitVector, string>, BitVector>();

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
            this.engine = engine;

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
                    Print.DebugMessage("Thread 1 local  ID = " + string.Join(", ", new List<BitVector>(localID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 1 global ID = " + string.Join(", ", new List<BitVector>(globalID1).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 local  ID = " + string.Join(", ", new List<BitVector>(localID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    Print.DebugMessage("Thread 2 global ID = " + string.Join(", ", new List<BitVector>(globalID2).ConvertAll(i => i.ToString()).ToArray()), 1);
                    EvaluateConstants(program.TopLevelDeclarations.OfType<Constant>());
                    InterpretKernel(program, impl);
                    executions++;
                }
                while (globalHeaderCount < engine.LoopHeaderLimit
                    && !AllBlocksCovered(impl) && executions < 5);

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
            Tuple<BitVector, BitVector> dimX = GetID(gpu.BlockDim[DIMENSION.X] - 1);
            Tuple<BitVector, BitVector> dimY = GetID(gpu.BlockDim[DIMENSION.Y] - 1);
            Tuple<BitVector, BitVector> dimZ = GetID(gpu.BlockDim[DIMENSION.Z] - 1);
            localID1[0] = dimX.Item1;
            localID2[0] = dimX.Item2;
            localID1[1] = dimY.Item1;
            localID2[1] = dimY.Item2;
            localID1[2] = dimZ.Item1;
            localID2[2] = dimZ.Item2;
        }

        private void SetGlobalIDs()
        {
            Tuple<BitVector, BitVector> dimX = GetID(gpu.GridDim[DIMENSION.X] - 1);
            Tuple<BitVector, BitVector> dimY = GetID(gpu.GridDim[DIMENSION.Y] - 1);
            Tuple<BitVector, BitVector> dimZ = GetID(gpu.GridDim[DIMENSION.Z] - 1);
            globalID1[0] = dimX.Item1;
            globalID2[0] = dimX.Item2;
            globalID1[1] = dimY.Item1;
            globalID2[1] = dimY.Item2;
            globalID1[2] = dimZ.Item1;
            globalID2[2] = dimZ.Item2;
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
                        if (binary.Op == "==")
                        {
                            if (binary.GetChildren()[0] is ScalarSymbolNode && binary.GetChildren()[1] is LiteralNode)
                            {
                                // Assume that equality is actually assignment into the variable of interest
                                search = false;
                                ScalarSymbolNode left = (ScalarSymbolNode)binary.GetChildren()[0];
                                LiteralNode right = (LiteralNode)binary.GetChildren()[1];
                                if (left.Symbol == "group_size_x")
                                {
                                    gpu.BlockDim[DIMENSION.X] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.X]));
                                }
                                else if (left.Symbol == "group_size_y")
                                {
                                    gpu.BlockDim[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.Y]));
                                }
                                else if (left.Symbol == "group_size_z")
                                {
                                    gpu.BlockDim[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.BlockDim[DIMENSION.Z]));
                                }
                                else if (left.Symbol == "num_groups_x")
                                {
                                    gpu.GridDim[DIMENSION.X] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.X]));
                                }
                                else if (left.Symbol == "num_groups_y")
                                {
                                    gpu.GridDim[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.Y]));
                                }
                                else if (left.Symbol == "num_groups_z")
                                {
                                    gpu.GridDim[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridDim[DIMENSION.Z]));
                                }
                                else if (left.Symbol == "global_offset_x")
                                {
                                    gpu.GridOffset[DIMENSION.X] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.X]));
                                }
                                else if (left.Symbol == "global_offset_y")
                                {
                                    gpu.GridOffset[DIMENSION.Y] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.Y]));
                                }
                                else if (left.Symbol == "global_offset_z")
                                {
                                    gpu.GridOffset[DIMENSION.Z] = right.Evaluation.ConvertToInt32();
                                    memory.Store(left.Symbol, new BitVector(gpu.GridOffset[DIMENSION.Z]));
                                }
                                else
                                {
                                    memory.Store(left.Symbol, right.Evaluation);
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
                if (RegularExpressions.TrackingVariable.IsMatch(decl.Name))
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
                    memory.Store(constant.Name, localID1[0]);
                }
                else if (constant.Name.Equals("local_id_y$1"))
                {
                    memory.Store(constant.Name, localID1[1]);
                }
                else if (constant.Name.Equals("local_id_z$1"))
                {
                    memory.Store(constant.Name, localID1[2]);
                }
                else if (constant.Name.Equals("local_id_x$2"))
                {
                    memory.Store(constant.Name, localID2[0]);
                }
                else if (constant.Name.Equals("local_id_y$2"))
                {
                    memory.Store(constant.Name, localID2[1]);
                }
                else if (constant.Name.Equals("local_id_z$2"))
                {
                    memory.Store(constant.Name, localID2[2]);
                }
                else if (constant.Name.Equals("group_id_x$1"))
                {
                    memory.Store(constant.Name, globalID1[0]);
                }
                else if (constant.Name.Equals("group_id_y$1"))
                {
                    memory.Store(constant.Name, globalID1[1]);
                }
                else if (constant.Name.Equals("group_id_z$1"))
                {
                    memory.Store(constant.Name, globalID1[2]);
                }
                else if (constant.Name.Equals("group_id_x$2"))
                {
                    memory.Store(constant.Name, globalID2[0]);
                }
                else if (constant.Name.Equals("group_id_y$2"))
                {
                    memory.Store(constant.Name, globalID2[1]);
                }
                else if (constant.Name.Equals("group_id_z$2"))
                {
                    memory.Store(constant.Name, globalID2[2]);
                }
                else if (constant.Name.Equals("group_id_x"))
                {
                    memory.Store(constant.Name, globalID1[0]);
                }
                else if (constant.Name.Equals("group_id_y"))
                {
                    memory.Store(constant.Name, globalID1[1]);
                }
                else if (constant.Name.Equals("group_id_z"))
                {
                    memory.Store(constant.Name, globalID1[2]);
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
                while (block != null
                    && globalHeaderCount < engine.LoopHeaderLimit
                    && stopwatch.Elapsed.Seconds < engine.TimeLimit)
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
                        if (scalar.Type.IsBv)
                        {
                            if (scalar.Type is BvType)
                            {
                                BvType bv = scalar.Type as BvType;
                                if (bv.Bits == 1)
                                {
                                    memory.Store(scalar.Symbol, BitVector.True);
                                }
                            }
                        }
                        else if (scalar.Type is BasicType)
                        {
                            BasicType basic = scalar.Type as BasicType;
                            if (basic.IsBool)
                                memory.Store(scalar.Symbol, BitVector.True);
                        }
                    }
                    else if (node is UnaryNode)
                    {
                        UnaryNode unary = node as UnaryNode;
                        ExprNode child = unary.GetChildren()[0] as ExprNode;
                        if (unary.Op == "!" && child is ScalarSymbolNode)
                        {
                            ScalarSymbolNode scalarChild = child as ScalarSymbolNode;
                            if (scalarChild.Type.IsBv)
                            {
                                BvType bv = scalarChild.Type as BvType;
                                if (bv.Bits == 1)
                                    memory.Store(scalarChild.Symbol, BitVector.False);
                            }
                            else if (scalarChild.Type is BasicType)
                            {
                                BasicType basic = scalarChild.Type as BasicType;
                                if (basic.IsBool)
                                    memory.Store(scalarChild.Symbol, BitVector.False);
                            }
                        }
                    }
                    else if (node is BinaryNode)
                    {
                        BinaryNode binary = node as BinaryNode;
                        if (binary.Op == "==")
                        {
                            ExprNode left = binary.GetChildren()[0] as ExprNode;
                            ExprNode right = binary.GetChildren()[1] as ExprNode;
                            if (right.Initialised)
                            {
                                if (left is ScalarSymbolNode)
                                {
                                    ScalarSymbolNode leftScalar = left as ScalarSymbolNode;
                                    memory.Store(leftScalar.Symbol, right.Evaluation);
                                }
                                else if (left is MapSymbolNode)
                                {
                                    MapSymbolNode leftMap = left as MapSymbolNode;
                                    Memory.SubscriptExpr subscriptExpr = new Memory.SubscriptExpr();
                                    foreach (ExprNode child in leftMap.GetChildren())
                                        subscriptExpr.Indices.Add(child.Evaluation);
                                    memory.Store(leftMap.Basename, subscriptExpr, right.Evaluation);
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
                            Memory.SubscriptExpr subscriptExpr = new Memory.SubscriptExpr();
                            foreach (Expr index in lhs.Indexes)
                            {
                                ExprTree tree = GetExprTree(index);
                                EvaluateExprTree(tree);
                                if (tree.Initialised)
                                    subscriptExpr.Indices.Add(tree.Evaluation);
                            }

                            if (subscriptExpr.Indices.Count > 0)
                            {
                                ExprTree tree2 = lhsEval.Item2;
                                if (tree2.Initialised)
                                    memory.Store(lhs.DeepAssignedVariable.Name, subscriptExpr, tree2.Evaluation);
                            }
                        }
                        else
                        {
                            SimpleAssignLhs lhs = (SimpleAssignLhs)lhsEval.Item1;
                            ExprTree tree = lhsEval.Item2;
                            if (tree.Initialised)
                                memory.Store(lhs.AssignedVariable.Name, tree.Evaluation);
                        }
                    }
                }
                else if (cmd is CallCmd)
                {
                    CallCmd call = cmd as CallCmd;
                    if (RegularExpressions.LogRead.IsMatch(call.callee))
                        LogRead(call);
                    else if (RegularExpressions.LogWrite.IsMatch(call.callee))
                        LogWrite(call);
                    else if (RegularExpressions.LogAtomic.IsMatch(call.callee))
                        LogAtomic(call);
                    else if (RegularExpressions.BugleBarrier.IsMatch(call.callee))
                        Barrier(call);
                }
                else if (cmd is AssertCmd)
                {
                    AssertCmd assert = cmd as AssertCmd;

                    // Only check asserts which have attributes as these are the candidate invariants
                    string tag = QKeyValue.FindStringAttribute(assert.Attributes, "tag");
                    if (tag != null)
                    {
                        MatchCollection matches = RegularExpressions.InvariantVariable.Matches(assert.ToString());
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
                                if (tree.OffsetVariables.Count > 0)
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
                    if (tree.Initialised && tree.Evaluation.Equals(BitVector.False))
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
            foreach (string offsetVariable in tree.OffsetVariables)
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
            if (tree.Initialised && tree.Evaluation.Equals(BitVector.False))
            {
                Print.VerboseMessage("==========> FALSE " + assert.ToString());
                assertStatus[assertBoolean] = BitVector.False;

                // Tell Houdini about the killed assert
                Houdini.RefutedAnnotation annotation = GPUVerify.Utilities.GetRefutedAnnotation(program, assertBoolean, impl.Name);
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
                        if (exprTree.Evaluation.Equals(BitVector.True))
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
            return binary.Op.Equals(BinaryOps.IF)
                || binary.Op.Equals(BinaryOps.IFF)
                || binary.Op.Equals(BinaryOps.AND)
                || binary.Op.Equals(BinaryOps.OR)
                || binary.Op.Equals(BinaryOps.NEQ)
                || binary.Op.Equals(BinaryOps.EQ)
                || binary.Op.Equals(BinaryOps.LT)
                || binary.Op.Equals(BinaryOps.LTE)
                || binary.Op.Equals(BinaryOps.GT)
                || binary.Op.Equals(BinaryOps.GTE)
                || RegularExpressions.BvSLT.IsMatch(binary.Op)
                || RegularExpressions.BvSLE.IsMatch(binary.Op)
                || RegularExpressions.BvSGT.IsMatch(binary.Op)
                || RegularExpressions.BvSGE.IsMatch(binary.Op)
                || RegularExpressions.BvULT.IsMatch(binary.Op)
                || RegularExpressions.BvULE.IsMatch(binary.Op)
                || RegularExpressions.BvUGT.IsMatch(binary.Op)
                || RegularExpressions.BvUGE.IsMatch(binary.Op);
        }

        private void EvaluateBinaryBoolNode(BinaryNode binary)
        {
            ExprNode left = binary.GetChildren()[0] as ExprNode;
            ExprNode right = binary.GetChildren()[1] as ExprNode;

            binary.Initialised = left.Initialised && right.Initialised;
            if (binary.Initialised)
            {
                if (binary.Op.Equals(BinaryOps.IF))
                {
                    if (left.Evaluation.Equals(BitVector.True) && right.Evaluation.Equals(BitVector.False))
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.EQ))
                {
                    if (left.Evaluation != right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.AND))
                {
                    if (!(left.Evaluation.Equals(BitVector.True) && right.Evaluation.Equals(BitVector.True)))
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.OR))
                {
                    if (left.Evaluation.Equals(BitVector.True) || right.Evaluation.Equals(BitVector.True))
                    {
                        binary.Evaluation = BitVector.True;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.False;
                    }
                }
                else if (RegularExpressions.BvSLT.IsMatch(binary.Op))
                {
                    if (left.Evaluation >= right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvSLE.IsMatch(binary.Op))
                {
                    if (left.Evaluation > right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.IFF))
                {
                    if ((left.Evaluation.Equals(BitVector.True) && right.Evaluation.Equals(BitVector.False))
                        || (left.Evaluation.Equals(BitVector.False) && right.Evaluation.Equals(BitVector.True)))
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.LT))
                {
                    if (left.Evaluation >= right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.LTE))
                {
                    if (left.Evaluation > right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.GT))
                {
                    if (left.Evaluation <= right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.GTE))
                {
                    if (left.Evaluation < right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (binary.Op.Equals(BinaryOps.NEQ))
                {
                    if (left.Evaluation == right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvSGT.IsMatch(binary.Op))
                {
                    if (left.Evaluation <= right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvSGE.IsMatch(binary.Op))
                {
                    if (left.Evaluation < right.Evaluation)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvULT.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    if (lhsUnsigned >= rhsUnsigned)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvULE.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    if (lhsUnsigned > rhsUnsigned)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvUGT.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    if (lhsUnsigned <= rhsUnsigned)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else if (RegularExpressions.BvUGE.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    if (lhsUnsigned < rhsUnsigned)
                    {
                        binary.Evaluation = BitVector.False;
                    }
                    else
                    {
                        binary.Evaluation = BitVector.True;
                    }
                }
                else
                {
                    throw new UnhandledException("Unhandled bv binary op: " + binary.Op);
                }
            }
        }

        private void EvaluateBinaryNonBoolNode(BinaryNode binary)
        {
            ExprNode left = binary.GetChildren()[0] as ExprNode;
            ExprNode right = binary.GetChildren()[1] as ExprNode;

            binary.Initialised = left.Initialised && right.Initialised;
            if (binary.Initialised)
            {
                if (RegularExpressions.BvADD.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation + right.Evaluation;
                }
                else if (RegularExpressions.BvSUB.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation - right.Evaluation;
                }
                else if (RegularExpressions.BvMUL.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation * right.Evaluation;
                }
                else if (RegularExpressions.BvAND.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation & right.Evaluation;
                }
                else if (RegularExpressions.BvOR.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation | right.Evaluation;
                }
                else if (binary.Op.Equals(BinaryOps.ADD))
                {
                    binary.Evaluation = left.Evaluation + right.Evaluation;
                }
                else if (binary.Op.Equals(BinaryOps.SUBTRACT))
                {
                    binary.Evaluation = left.Evaluation - right.Evaluation;
                }
                else if (binary.Op.Equals(BinaryOps.MULTIPLY))
                {
                    binary.Evaluation = left.Evaluation * right.Evaluation;
                }
                else if (binary.Op.Equals(BinaryOps.DIVIDE))
                {
                    binary.Evaluation = left.Evaluation / right.Evaluation;
                }
                else if (RegularExpressions.BvASHR.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation >> right.Evaluation.ConvertToInt32();
                }
                else if (RegularExpressions.BvLSHR.IsMatch(binary.Op))
                {
                    binary.Evaluation = BitVector.LogicalShiftRight(left.Evaluation, right.Evaluation.ConvertToInt32());
                }
                else if (RegularExpressions.BvSHL.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation << right.Evaluation.ConvertToInt32();
                }
                else if (RegularExpressions.BvDIV.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation / right.Evaluation;
                }
                else if (RegularExpressions.BvXOR.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation ^ right.Evaluation;
                }
                else if (RegularExpressions.BvSREM.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation % right.Evaluation;
                }
                else if (RegularExpressions.BvUREM.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    binary.Evaluation = lhsUnsigned % rhsUnsigned;
                }
                else if (RegularExpressions.BvSDIV.IsMatch(binary.Op))
                {
                    binary.Evaluation = left.Evaluation / right.Evaluation;
                }
                else if (RegularExpressions.BvUDIV.IsMatch(binary.Op))
                {
                    BitVector lhsUnsigned = left.Evaluation >= BitVector.Zero(left.Evaluation.Bits.Length)
                        ? left.Evaluation : left.Evaluation & BitVector.Max(left.Evaluation.Bits.Length);
                    BitVector rhsUnsigned = right.Evaluation >= BitVector.Zero(right.Evaluation.Bits.Length)
                        ? right.Evaluation : right.Evaluation & BitVector.Max(right.Evaluation.Bits.Length);

                    binary.Evaluation = lhsUnsigned / rhsUnsigned;
                }
                else if (binary.Op.Equals("FEQ32")
                    || binary.Op.Equals("FEQ64")
                    || binary.Op.Equals("FGE32")
                    || binary.Op.Equals("FGE64")
                    || binary.Op.Equals("FGT32")
                    || binary.Op.Equals("FGT64")
                    || binary.Op.Equals("FLE32")
                    || binary.Op.Equals("FLE64")
                    || binary.Op.Equals("FLT32")
                    || binary.Op.Equals("FLT64")
                    || binary.Op.Equals("FUNO32")
                    || binary.Op.Equals("FUNO64"))
                {
                    Tuple<BitVector, BitVector, string> fpTriple = Tuple.Create(left.Evaluation, right.Evaluation, binary.Op);

                    if (!fpInterpretations.ContainsKey(fpTriple))
                    {
                        if (random.Next(0, 2) == 0)
                            fpInterpretations[fpTriple] = BitVector.False;
                        else
                            fpInterpretations[fpTriple] = BitVector.True;
                    }

                    binary.Evaluation = fpInterpretations[fpTriple];
                }
                else if (binary.Op.Equals("FADD32")
                    || binary.Op.Equals("FSUB32")
                    || binary.Op.Equals("FMUL32")
                    || binary.Op.Equals("FDIV32")
                    || binary.Op.Equals("FPOW32"))
                {
                    Tuple<BitVector, BitVector, string> fpTriple = Tuple.Create(left.Evaluation, right.Evaluation, binary.Op);

                    if (!fpInterpretations.ContainsKey(fpTriple))
                        fpInterpretations[fpTriple] = new BitVector(random.Next(), 32);

                    binary.Evaluation = fpInterpretations[fpTriple];
                }
                else if (binary.Op.Equals("FADD64")
                    || binary.Op.Equals("FSUB64")
                    || binary.Op.Equals("FMUL64")
                    || binary.Op.Equals("FDIV64")
                    || binary.Op.Equals("FPOW64"))
                {
                    Tuple<BitVector, BitVector, string> fpTriple = Tuple.Create(left.Evaluation, right.Evaluation, binary.Op);

                    if (!fpInterpretations.ContainsKey(fpTriple))
                        fpInterpretations[fpTriple] = new BitVector(random.Next(), 64);

                    binary.Evaluation = fpInterpretations[fpTriple];
                }
                else
                {
                    throw new UnhandledException("Unhandled bv binary op: " + binary.Op);
                }
            }
        }

        private void EvaluateUnaryNode(UnaryNode unary)
        {
            ExprNode child = unary.GetChildren()[0] as ExprNode;
            unary.Initialised = child.Initialised;
            if (unary.Initialised)
            {
                if (unary.Op.Equals("!"))
                {
                    if (child.Evaluation.Equals(BitVector.True))
                        unary.Evaluation = BitVector.False;
                    else
                        unary.Evaluation = BitVector.True;
                }
                else if (unary.Op.Equals("FABS32")
                    || unary.Op.Equals("FCOS32")
                    || unary.Op.Equals("FEXP32")
                    || unary.Op.Equals("FFLOOR32")
                    || unary.Op.Equals("FLOG32")
                    || unary.Op.Equals("FPOW32")
                    || unary.Op.Equals("FSIN32")
                    || unary.Op.Equals("FSQRT32"))
                {
                    Tuple<BitVector, BitVector, string> fpTriple = Tuple.Create(child.Evaluation, child.Evaluation, unary.Op);

                    if (!fpInterpretations.ContainsKey(fpTriple))
                        fpInterpretations[fpTriple] = new BitVector(random.Next(), 32);

                    unary.Evaluation = fpInterpretations[fpTriple];
                }
                else if (unary.Op.Equals("FABS64")
                    || unary.Op.Equals("FCOS64")
                    || unary.Op.Equals("FEXP64")
                    || unary.Op.Equals("FFLOOR64")
                    || unary.Op.Equals("FLOG64")
                    || unary.Op.Equals("FPOW64")
                    || unary.Op.Equals("FSIN64")
                    || unary.Op.Equals("FSQRT64"))
                {
                    Tuple<BitVector, BitVector, string> fpTriple = Tuple.Create(child.Evaluation, child.Evaluation, unary.Op);

                    if (!fpInterpretations.ContainsKey(fpTriple))
                        fpInterpretations[fpTriple] = new BitVector(random.Next(), 64);

                    unary.Evaluation = fpInterpretations[fpTriple];
                }
                else if (RegularExpressions.BvZEXT.IsMatch(unary.Op))
                {
                    int width = 32;
                    MatchCollection matches = Regex.Matches(unary.Op, @"\d+");

                    if (matches.Count == 2)
                        width = Convert.ToInt32(matches[1].Value);

                    unary.Evaluation = BitVector.ZeroExtend(child.Evaluation, width);
                }
                else if (RegularExpressions.BvSEXT.IsMatch(unary.Op))
                {
                    int width = 32;
                    MatchCollection matches = Regex.Matches(unary.Op, @"\d+");

                    if (matches.Count == 2)
                        width = Convert.ToInt32(matches[1].Value);

                    unary.Evaluation = BitVector.SignExtend(child.Evaluation, width);
                }
                else if (RegularExpressions.CastToFP.IsMatch(unary.Op))
                {
                    MatchCollection matches = Regex.Matches(unary.Op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);

                    if (sourceSize == destinationSize)
                        unary.Evaluation = child.Evaluation;
                    else if (sourceSize > destinationSize)
                        unary.Evaluation = BitVector.Slice(child.Evaluation, destinationSize, 0);
                    else
                        unary.Evaluation = BitVector.ZeroExtend(child.Evaluation, destinationSize);
                }
                else if (RegularExpressions.CastToInt.IsMatch(unary.Op))
                {
                    MatchCollection matches = Regex.Matches(unary.Op, @"\d+");
                    Debug.Assert(matches.Count == 2);
                    int sourceSize = Convert.ToInt32(matches[0].Value);
                    int destinationSize = Convert.ToInt32(matches[1].Value);

                    if (sourceSize == destinationSize)
                        unary.Evaluation = child.Evaluation;
                    else if (sourceSize > destinationSize)
                        unary.Evaluation = BitVector.Slice(child.Evaluation, destinationSize, 0);
                    else
                        unary.Evaluation = BitVector.ZeroExtend(child.Evaluation, destinationSize);
                }
                else if (RegularExpressions.CastFPToDouble.IsMatch(unary.Op))
                {
                    unary.Evaluation = BitVector.ZeroExtend(child.Evaluation, 32);
                }
                else if (unary.Op.Equals("FUNCPTR_TO_PTR")
                    || unary.Op.Equals("PTR_TO_FUNCPTR"))
                {
                    unary.Evaluation = child.Evaluation;
                }
                else
                {
                    throw new UnhandledException("Unhandled bv unary op: " + unary.Op);
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
                        ScalarSymbolNode scalarNode = node as ScalarSymbolNode;
                        if (memory.Contains(scalarNode.Symbol))
                            scalarNode.Evaluation = memory.GetValue(scalarNode.Symbol);
                        else
                            scalarNode.Initialised = false;
                    }
                    else if (node is MapSymbolNode)
                    {
                        MapSymbolNode mapNode = node as MapSymbolNode;
                        Memory.SubscriptExpr subscriptExpr = new Memory.SubscriptExpr();
                        foreach (ExprNode child in mapNode.GetChildren())
                        {
                            if (child.Initialised)
                                subscriptExpr.Indices.Add(child.Evaluation);
                            else
                                mapNode.Initialised = false;
                        }

                        if (node.Initialised)
                        {
                            if (memory.Contains(mapNode.Basename, subscriptExpr))
                                mapNode.Evaluation = memory.GetValue(mapNode.Basename, subscriptExpr);
                            else
                                mapNode.Initialised = false;
                        }
                    }
                    else if (node is BVExtractNode)
                    {
                        BVExtractNode extractNode = node as BVExtractNode;
                        ExprNode child = (ExprNode)extractNode.GetChildren()[0];
                        if (child.Initialised)
                            extractNode.Evaluation = BitVector.Slice(child.Evaluation, extractNode.High, extractNode.Low);
                        else
                            extractNode.Initialised = false;
                    }
                    else if (node is BVConcatenationNode)
                    {
                        BVConcatenationNode concatNode = node as BVConcatenationNode;
                        ExprNode one = (ExprNode)concatNode.GetChildren()[0];
                        ExprNode two = (ExprNode)concatNode.GetChildren()[1];
                        if (one.Initialised && two.Initialised)
                            concatNode.Evaluation = BitVector.Concatenate(one.Evaluation, two.Evaluation);
                        else
                            concatNode.Initialised = false;
                    }
                    else if (node is UnaryNode)
                    {
                        UnaryNode unaryNode = node as UnaryNode;
                        EvaluateUnaryNode(unaryNode);
                    }
                    else if (node is BinaryNode)
                    {
                        BinaryNode binaryNode = node as BinaryNode;
                        if (IsBoolBinaryOp(binaryNode))
                            EvaluateBinaryBoolNode(binaryNode);
                        else
                            EvaluateBinaryNonBoolNode(binaryNode);
                    }
                    else if (node is TernaryNode)
                    {
                        TernaryNode ternaryNode = node as TernaryNode;
                        ExprNode one = (ExprNode)ternaryNode.GetChildren()[0];
                        ExprNode two = (ExprNode)ternaryNode.GetChildren()[1];
                        ExprNode three = (ExprNode)ternaryNode.GetChildren()[2];
                        if (!one.Initialised)
                        {
                            ternaryNode.Initialised = false;
                        }
                        else
                        {
                            if (one.Evaluation.Equals(BitVector.True))
                            {
                                if (two.Initialised)
                                    ternaryNode.Evaluation = two.Evaluation;
                                else
                                    ternaryNode.Initialised = false;
                            }
                            else
                            {
                                if (three.Initialised)
                                    ternaryNode.Evaluation = three.Evaluation;
                                else
                                    ternaryNode.Initialised = false;
                            }
                        }
                    }
                }
            }

            ExprNode root = tree.Root() as ExprNode;
            tree.Initialised = root.Initialised;
            tree.Evaluation = root.Evaluation;
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
                if ((memory.IsInGlobalMemory(arrayName) && globalTree.Evaluation.Equals(BitVector.True))
                    || (memory.IsInGroupSharedMemory(arrayName) && groupSharedTree.Evaluation.Equals(BitVector.True)))
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
                                }

                                break;
                            case "_READ_OFFSET_":
                                {
                                    string accessTracker = "_READ_HAS_OCCURRED_" + arrayName;
                                    memory.Store(accessTracker, BitVector.False);
                                }

                                break;
                            case "_ATOMIC_OFFSET_":
                                {
                                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName;
                                    memory.Store(accessTracker, BitVector.False);
                                }

                                break;
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
            if (tree1.Initialised && tree1.Evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.Initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.Evaluation);
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
            if (tree1.Initialised && tree1.Evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.Initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.Evaluation);
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
            if (tree1.Initialised && tree1.Evaluation.Equals(BitVector.True))
            {
                ExprTree tree2 = GetExprTree(call.Ins[1]);
                EvaluateExprTree(tree2);
                if (tree2.Initialised)
                {
                    memory.AddRaceArrayOffset(raceArrayOffsetName, tree2.Evaluation);
                    string accessTracker = "_ATOMIC_HAS_OCCURRED_" + arrayName;
                    memory.Store(accessTracker, BitVector.True);
                }
            }
        }

        [Serializable]
        public class UnhandledException : Exception
        {
            public UnhandledException(string message)
                 : base(message)
            {
            }
        }

        [Serializable]
        public class TimeLimitException : Exception
        {
            public TimeLimitException(string message)
                 : base(message)
            {
            }
        }

        public static class RegularExpressions
        {
            public static readonly Regex InvariantVariable = new Regex("_[a-z][0-9]+");

            // Case sensitive
            public static readonly Regex WatchdoVariable = new Regex("_WATCHED_OFFSET", RegexOptions.IgnoreCase);
            public static readonly Regex OffsetVariable = new Regex("_(WRITE|READ|ATOMIC)_OFFSET_", RegexOptions.IgnoreCase);
            public static readonly Regex TrackingVariable = new Regex("_(WRITE|READ|ATOMIC)_HAS_OCCURRED_", RegexOptions.IgnoreCase);
            public static readonly Regex LogRead = new Regex("_LOG_READ_", RegexOptions.IgnoreCase);
            public static readonly Regex LogWrite = new Regex("_LOG_WRITE_", RegexOptions.IgnoreCase);
            public static readonly Regex LogAtomic = new Regex("_LOG_ATOMIC_", RegexOptions.IgnoreCase);
            public static readonly Regex BugleBarrier = new Regex("bugle_barrier", RegexOptions.IgnoreCase);
            public static readonly Regex BvSLE = new Regex("BV[0-9]+_SLE", RegexOptions.IgnoreCase);
            public static readonly Regex BvSLT = new Regex("BV[0-9]+_SLT", RegexOptions.IgnoreCase);
            public static readonly Regex BvSGE = new Regex("BV[0-9]+_SGE", RegexOptions.IgnoreCase);
            public static readonly Regex BvSGT = new Regex("BV[0-9]+_SGT", RegexOptions.IgnoreCase);
            public static readonly Regex BvULE = new Regex("BV[0-9]+_ULE", RegexOptions.IgnoreCase);
            public static readonly Regex BvULT = new Regex("BV[0-9]+_ULT", RegexOptions.IgnoreCase);
            public static readonly Regex BvUGE = new Regex("BV[0-9]+_UGE", RegexOptions.IgnoreCase);
            public static readonly Regex BvUGT = new Regex("BV[0-9]+_UGT", RegexOptions.IgnoreCase);
            public static readonly Regex BvASHR = new Regex("BV[0-9]+_ASHR", RegexOptions.IgnoreCase);
            public static readonly Regex BvLSHR = new Regex("BV[0-9]+_LSHR", RegexOptions.IgnoreCase);
            public static readonly Regex BvSHL = new Regex("BV[0-9]+_SHL", RegexOptions.IgnoreCase);
            public static readonly Regex BvADD = new Regex("BV[0-9]+_ADD", RegexOptions.IgnoreCase);
            public static readonly Regex BvSUB = new Regex("BV[0-9]+_SUB", RegexOptions.IgnoreCase);
            public static readonly Regex BvMUL = new Regex("BV[0-9]+_MUL", RegexOptions.IgnoreCase);
            public static readonly Regex BvDIV = new Regex("BV[0-9]+_DIV", RegexOptions.IgnoreCase);
            public static readonly Regex BvAND = new Regex("BV[0-9]+_AND", RegexOptions.IgnoreCase);
            public static readonly Regex BvOR = new Regex("BV[0-9]+_OR", RegexOptions.IgnoreCase);
            public static readonly Regex BvXOR = new Regex("BV[0-9]+_XOR", RegexOptions.IgnoreCase);
            public static readonly Regex BvSREM = new Regex("BV[0-9]+_SREM", RegexOptions.IgnoreCase);
            public static readonly Regex BvUREM = new Regex("BV[0-9]+_UREM", RegexOptions.IgnoreCase);
            public static readonly Regex BvSDIV = new Regex("BV[0-9]+_SDIV", RegexOptions.IgnoreCase);
            public static readonly Regex BvUDIV = new Regex("BV[0-9]+_UDIV", RegexOptions.IgnoreCase);
            public static readonly Regex BvZEXT = new Regex("BV[0-9]+_ZEXT", RegexOptions.IgnoreCase);
            public static readonly Regex BvSEXT = new Regex("BV[0-9]+_SEXT", RegexOptions.IgnoreCase);
            public static readonly Regex CastToFP = new Regex("(U|S)I[0-9]+_TO_FP[0-9]+", RegexOptions.IgnoreCase);
            public static readonly Regex CastToInt = new Regex("FP[0-9]+_TO_(U|S)I[0-9]+", RegexOptions.IgnoreCase);
            public static readonly Regex CastFPToDouble = new Regex("FP[0-9]+_CONV[0-9]+", RegexOptions.IgnoreCase);
        }

        private static class BinaryOps
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
    }
}
