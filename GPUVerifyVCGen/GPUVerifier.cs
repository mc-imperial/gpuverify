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
    using System.Diagnostics.Contracts;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Numerics;
    using Microsoft.Basetypes;
    using Microsoft.Boogie;

    public class GPUVerifier : CheckingContext
    {
        private string outputFilename;
        private ResolutionContext resContext;

        public Program Program { get; private set; }

        public IntegerRepresentation IntRep { get; private set; }

        public Dictionary<Procedure, Implementation> KernelProcedures { get; private set; }

        public Dictionary<string, string> GlobalArraySourceNames { get; private set; }

        private HashSet<Procedure> barrierProcedures = new HashSet<Procedure>();
        private Dictionary<Tuple<Variable, AccessType, bool>, Procedure> warpSyncs = new Dictionary<Tuple<Variable, AccessType, bool>, Procedure>();
        private string barrierProcedureLocalFenceArgName;
        private string barrierProcedureGlobalFenceArgName;

        private HashSet<object> regionsWithLoopInvariantsDisabled = new HashSet<object>();

        public IKernelArrayInfo KernelArrayInfo { get; } = new KernelArrayInfoLists();

        private HashSet<string> reservedNames = new HashSet<string>();

        public const string SizeTBitsTypeString = "_SIZE_T_TYPE";

        public Microsoft.Boogie.Type SizeTType { get; }

        public Microsoft.Boogie.Type IdType { get; }

        public HashSet<string> OnlyThread1 { get; } = new HashSet<string>();

        public HashSet<string> OnlyThread2 { get; } = new HashSet<string>();

        private const string LocalIdXString = "local_id_x";
        private const string LocalIdYString = "local_id_y";
        private const string LocalIdZString = "local_id_z";

        private Constant idX = null;
        private Constant idY = null;
        private Constant idZ = null;

        public Constant IdX => idX;

        public Constant IdY => idY;

        public Constant IdZ => idZ;

        private const string GroupSizeXString = "group_size_x";
        private const string GroupSizeYString = "group_size_y";
        private const string GroupSizeZString = "group_size_z";

        private Constant groupSizeX = null;
        private Constant groupSizeY = null;
        private Constant groupSizeZ = null;

        private const string GroupIdXString = "group_id_x";
        private const string GroupIdYString = "group_id_y";
        private const string GroupIdZString = "group_id_z";

        private Constant groupIdX = null;
        private Constant groupIdY = null;
        private Constant groupIdZ = null;

        private const string NumGroupsXString = "num_groups_x";
        private const string NumGroupsYString = "num_groups_y";
        private const string NumGroupsZString = "num_groups_z";

        private Constant numGroupsX = null;
        private Constant numGroupsY = null;
        private Constant numGroupsZ = null;

        private const string SubGroupSizeString = "sub_group_size";

        private Constant subGroupSize = null;

        public IRaceInstrumenter RaceInstrumenter { get; }

        public INoAccessInstrumenter NoAccessInstrumenter { get; }

        public IConstantWriteInstrumenter ConstantWriteInstrumenter { get; }

        public UniformityAnalyser UniformityAnalyser { get; private set; }

        public MayBePowerOfTwoAnalyser MayBePowerOfTwoAnalyser { get; private set; }

        public RelationalPowerOfTwoAnalyser RelationalPowerOfTwoAnalyser { get; private set; }

        private ArrayControlFlowAnalyser arrayControlFlowAnalyser;
        private CallSiteAnalyser callSiteAnalyser;

        public Dictionary<Implementation, VariableDefinitionAnalysisRegion> VarDefAnalysesRegion { get; private set; }

        public Dictionary<Implementation, ReducedStrengthAnalysisRegion> ReducedStrengthAnalysesRegion { get; private set; }

        public Dictionary<AccessType, HashSet<string>> ArraysAccessedByAsyncWorkGroupCopy { get; }

        public GPUVerifier(string filename, Program program, ResolutionContext rc)
            : base(null)
        {
            this.outputFilename = filename;
            this.Program = program;
            this.resContext = rc;
            this.IntRep = GPUVerifyVCGenCommandLineOptions.MathInt
                ? (IntegerRepresentation)new MathIntegerRepresentation(this)
                : (IntegerRepresentation)new BVIntegerRepresentation(this);

            this.SizeTType = GetSizeTType();
            this.IdType = GetIdType();

            this.ArraysAccessedByAsyncWorkGroupCopy = new Dictionary<AccessType, HashSet<string>>();
            this.ArraysAccessedByAsyncWorkGroupCopy[AccessType.READ] = new HashSet<string>();
            this.ArraysAccessedByAsyncWorkGroupCopy[AccessType.WRITE] = new HashSet<string>();

            if (this.SizeTType.IsBv && this.SizeTType.BvBits < this.IdType.BvBits)
            {
                Console.WriteLine("GPUVerify: error: _SIZE_T_TYPE size cannot be smaller than group_size_x size");
                Environment.Exit(1);
            }

            new ModSetCollector().DoModSetAnalysis(Program);

            GlobalArraySourceNames = new Dictionary<string, string>();
            foreach (var g in Program.TopLevelDeclarations.OfType<GlobalVariable>())
            {
                string sourceName = QKeyValue.FindStringAttribute(g.Attributes, "source_name");
                if (sourceName != null)
                {
                    GlobalArraySourceNames[g.Name] = sourceName;
                }
                else
                {
                    GlobalArraySourceNames[g.Name] = g.Name;
                }
            }

            CheckWellFormedness();

            StripOutAnnotationsForDisabledArrays();

            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
            {
                this.NoAccessInstrumenter = new NoAccessInstrumenter(this);
            }

            this.ConstantWriteInstrumenter = new ConstantWriteInstrumenter(this);

            if (GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
            {
                this.RaceInstrumenter = new NullRaceInstrumenter();
            }
            else
            {
                if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.ORIGINAL)
                {
                    this.RaceInstrumenter = new OriginalRaceInstrumenter(this);
                }
                else
                {
                    Debug.Assert(RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE
                      || RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE);
                    this.RaceInstrumenter = new WatchdogRaceInstrumenter(this);
                }
            }
        }

        private void CheckWellFormedness()
        {
            int errorCount = Check();
            if (errorCount != 0)
            {
                Console.WriteLine(
                    "{0} GPUVerify format errors detected in {1}",
                    errorCount,
                    GPUVerifyVCGenCommandLineOptions.InputFiles[GPUVerifyVCGenCommandLineOptions.InputFiles.Count - 1]);
                Environment.Exit(1);
            }

            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
            {
                foreach (var b in Program.Blocks())
                {
                    foreach (var c in b.Cmds.OfType<CallCmd>())
                    {
                        if (QKeyValue.FindBoolAttribute(c.Attributes, "atomic"))
                        {
                            Console.WriteLine("GPUVerify: error: --equality-abstraction cannot be used with atomics.");
                            Environment.Exit(1);
                        }
                    }
                }
            }

            if (GPUVerifyVCGenCommandLineOptions.CheckSingleNonInlinedImpl)
            {
                var nonInlinedImpls = Program.Implementations
                    .Where(item => QKeyValue.FindIntAttribute((item as Implementation).Attributes, "inline", -1) == -1);
                if (nonInlinedImpls.Count() != 1)
                {
                    Console.WriteLine("GPUVerify: warning: Found {0} non-inlined implementations.", nonInlinedImpls.Count());
                    foreach (Implementation impl in nonInlinedImpls)
                        Console.WriteLine("  {0}", impl.Name);
                }
            }
        }

        private Microsoft.Boogie.Type GetSizeTType()
        {
            var candidates = Program.TopLevelDeclarations.OfType<TypeSynonymDecl>()
                .Where(item => item.Name == SizeTBitsTypeString);

            if (candidates.Count() != 1 || !candidates.Single().Body.IsBv)
            {
                Console.WriteLine("GPUVerify: error: exactly one _SIZE_T_TYPE bit-vector type must be specified");
                Environment.Exit(1);
            }

            // Do not use the size_t type directly, as we might be using
            // mathematical integers as our integer representation, and
            // as the size_t type is always a bitvector type.
            return IntRep.GetIntType(candidates.Single().Body.BvBits);
        }

        private Microsoft.Boogie.Type GetIdType()
        {
            var candidates = Program.TopLevelDeclarations.OfType<Constant>()
                .Where(item => item.Name == GroupSizeXString);

            if (candidates.Count() != 1)
            {
                Console.WriteLine("GPUVerify: error: exactly one group_size_x must be specified");
                Environment.Exit(1);
            }

            if (!candidates.Single().TypedIdent.Type.IsBv && !candidates.Single().TypedIdent.Type.IsInt)
            {
                Console.WriteLine("GPUVerify: error: group_size_x must be of type int or bv");
                Environment.Exit(1);
            }

            return candidates.Single().TypedIdent.Type;
        }

        public bool IsKernelProcedure(Procedure proc)
        {
            return QKeyValue.FindBoolAttribute(proc.Attributes, "kernel");
        }

        private Dictionary<Procedure, Implementation> GetKernelProcedures()
        {
            var result = new Dictionary<Procedure, Implementation>();
            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (QKeyValue.FindBoolAttribute(decl.Attributes, "kernel"))
                {
                    if (decl is Implementation)
                    {
                        result[(decl as Implementation).Proc] = decl as Implementation;
                    }

                    if (decl is Procedure)
                    {
                        if (!result.ContainsKey(decl as Procedure))
                        {
                            result[decl as Procedure] = null;
                        }
                    }
                }
            }

            return result;
        }

        private Procedure FindOrCreateBarrierProcedure()
        {
            var p = CheckSingleInstanceOfAttributedProcedure("barrier");
            if (p == null)
            {
                var inParams = new List<Variable>
                {
                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__local_fence", IntRep.GetIntType(1)), true),
                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__global_fence", IntRep.GetIntType(1)), true)
                };
                p = new Procedure(
                    Token.NoToken,
                    "barrier",
                    new List<TypeVariable>(),
                    inParams,
                    new List<Variable>(),
                    new List<Requires>(),
                    new List<IdentifierExpr>(),
                    new List<Ensures>(),
                    new QKeyValue(Token.NoToken, "barrier", new List<object>(), null));
                Program.AddTopLevelDeclaration(p);
                resContext.AddProcedure(p);
            }

            return p;
        }

        private Procedure CheckSingleInstanceOfAttributedProcedure(string attribute)
        {
            Procedure attributedProcedure = null;

            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (!QKeyValue.FindBoolAttribute(decl.Attributes, attribute))
                    continue;

                if (decl is Implementation)
                    continue;

                if (decl is Procedure)
                {
                    if (attributedProcedure == null)
                        attributedProcedure = decl as Procedure;
                    else
                        Error(decl, "\"{0}\" attribute specified for procedure {1}, but it has already been specified for procedure {2}", attribute, (decl as Procedure).Name, attributedProcedure.Name);
                }
                else
                {
                    Error(decl, "\"{0}\" attribute can only be applied to a procedure", attribute);
                }
            }

            return attributedProcedure;
        }

        private void ReportMultipleAttributeError(string attribute, IToken first, IToken second)
        {
            Error(
                second,
                "Can only have one {0} attribute, but previously saw this attribute at ({1}, {2})",
                attribute,
                first.filename,
                first.line);
        }

        private bool SetConstAttributeField(Constant constInProgram, string attr, ref Constant constFieldRef)
        {
            if (QKeyValue.FindBoolAttribute(constInProgram.Attributes, attr))
            {
                if (constFieldRef != null)
                {
                    ReportMultipleAttributeError(attr, constFieldRef.tok, constInProgram.tok);
                    return false;
                }

                CheckSpecialConstantType(constInProgram);
                constFieldRef = constInProgram;
            }

            return true;
        }

        private void MaybeCreateAttributedConst(string attr, ref Constant constFieldRef)
        {
            if (constFieldRef == null)
            {
                constFieldRef = new Constant(
                    Token.NoToken, new TypedIdent(Token.NoToken, attr, IdType), /*unique=*/false);
                constFieldRef.AddAttribute(attr);
                Program.AddTopLevelDeclaration(constFieldRef);
            }
        }

        private bool FindNonLocalVariables()
        {
            bool success = true;
            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (decl is Variable
                    && (decl as Variable).IsMutable
                    && (decl as Variable).TypedIdent.Type is MapType
                    && !reservedNames.Contains((decl as Variable).Name))
                {
                    if (QKeyValue.FindBoolAttribute(decl.Attributes, "group_shared"))
                    {
                        KernelArrayInfo.AddGroupSharedArray(decl as Variable);
                        if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null
                            && !GPUVerifyVCGenCommandLineOptions.ArraysToCheck.Contains(GlobalArraySourceNames[(decl as Variable).Name]))
                        {
                            KernelArrayInfo.DisableGlobalOrGroupSharedArray(decl as Variable);
                        }
                    }
                    else if (QKeyValue.FindBoolAttribute(decl.Attributes, "global"))
                    {
                        KernelArrayInfo.AddGlobalArray(decl as Variable);
                        if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null
                            && !GPUVerifyVCGenCommandLineOptions.ArraysToCheck.Contains(GlobalArraySourceNames[(decl as Variable).Name]))
                        {
                            KernelArrayInfo.DisableGlobalOrGroupSharedArray(decl as Variable);
                        }
                    }
                    else if (QKeyValue.FindBoolAttribute(decl.Attributes, "constant"))
                    {
                        KernelArrayInfo.AddConstantArray(decl as Variable);
                    }
                    else
                    {
                        if (!QKeyValue.FindBoolAttribute(decl.Attributes, "atomic_usedmap"))
                        {
                            KernelArrayInfo.AddPrivateArray(decl as Variable);
                        }
                    }
                }
                else if (decl is Constant)
                {
                    Constant c = decl as Constant;

                    success &= SetConstAttributeField(c, LocalIdXString, ref idX);
                    success &= SetConstAttributeField(c, LocalIdYString, ref idY);
                    success &= SetConstAttributeField(c, LocalIdZString, ref idZ);

                    success &= SetConstAttributeField(c, GroupSizeXString, ref groupSizeX);
                    success &= SetConstAttributeField(c, GroupSizeYString, ref groupSizeY);
                    success &= SetConstAttributeField(c, GroupSizeZString, ref groupSizeZ);

                    success &= SetConstAttributeField(c, GroupIdXString, ref groupIdX);
                    success &= SetConstAttributeField(c, GroupIdYString, ref groupIdY);
                    success &= SetConstAttributeField(c, GroupIdZString, ref groupIdZ);

                    success &= SetConstAttributeField(c, NumGroupsXString, ref numGroupsX);
                    success &= SetConstAttributeField(c, NumGroupsYString, ref numGroupsY);
                    success &= SetConstAttributeField(c, NumGroupsZString, ref numGroupsZ);

                    success &= SetConstAttributeField(c, SubGroupSizeString, ref subGroupSize);
                }
            }

            foreach (var c in Program.Blocks().SelectMany(item => item.Cmds).OfType<CallCmd>())
            {
                if (QKeyValue.FindBoolAttribute(c.Attributes, "atomic"))
                {
                    Debug.Assert(c.Ins.Count() >= 1);
                    var ie = c.Ins[0] as IdentifierExpr;
                    Debug.Assert(ie != null);
                    Debug.Assert(KernelArrayInfo.GetGlobalAndGroupSharedArrays(true).Contains(ie.Decl));
                    if (!KernelArrayInfo.GetAtomicallyAccessedArrays(true).Contains(ie.Decl))
                        KernelArrayInfo.AddAtomicallyAccessedArray(ie.Decl);
                }
            }

            MaybeCreateAttributedConst(LocalIdXString, ref idX);
            MaybeCreateAttributedConst(LocalIdYString, ref idY);
            MaybeCreateAttributedConst(LocalIdZString, ref idZ);

            MaybeCreateAttributedConst(GroupSizeXString, ref groupSizeX);
            MaybeCreateAttributedConst(GroupSizeYString, ref groupSizeY);
            MaybeCreateAttributedConst(GroupSizeZString, ref groupSizeZ);

            MaybeCreateAttributedConst(GroupIdXString, ref groupIdX);
            MaybeCreateAttributedConst(GroupIdYString, ref groupIdY);
            MaybeCreateAttributedConst(GroupIdZString, ref groupIdZ);

            MaybeCreateAttributedConst(NumGroupsXString, ref numGroupsX);
            MaybeCreateAttributedConst(NumGroupsYString, ref numGroupsY);
            MaybeCreateAttributedConst(NumGroupsZString, ref numGroupsZ);

            if (GPUVerifyVCGenCommandLineOptions.EliminateRedundantReadInstrumentation)
                ComputeReadOnlyArrays();

            return success;
        }

        private void ComputeReadOnlyArrays()
        {
            IEnumerable<Variable> writtenArrays =
             Program.TopLevelDeclarations.OfType<Procedure>()
                    .Select(item => item.Modifies)
                    .SelectMany(item => item)
                    .Select(item => item.Decl)
                    .Where(item => KernelArrayInfo.ContainsGlobalOrGroupSharedArray(item, true));
            foreach (var v in KernelArrayInfo.GetGlobalAndGroupSharedArrays(true)
                .Where(item => !writtenArrays.Contains(item)))
            {
                KernelArrayInfo.AddReadOnlyGlobalOrGroupSharedArray(v);
            }
        }

        private void CheckSpecialConstantType(Constant c)
        {
            if (!(c.TypedIdent.Type.IsInt || c.TypedIdent.Type.IsBv))
                Error(c.tok, "Special constant '" + c.Name + "' must have type 'int' or 'bv'");
        }

        private void MergeBlocksIntoPredecessors(bool uniformityMatters = true)
        {
            foreach (var impl in Program.Implementations)
            {
                UniformityAnalyser.MergeBlocksIntoPredecessors(
                    Program, impl, uniformityMatters ? UniformityAnalyser : null);
            }
        }

        public void DoIt()
        {
            CommandLineOptions.Clo.PrintUnstructured = 2;

            if (GPUVerifyVCGenCommandLineOptions.PrintLoopStatistics)
                PrintLoopStatistics();

            RemoveUnnecessaryBlockSourceLocations();

            PropagateProcedureWideInvariants();

            CheckUserSuppliedLoopInvariants();

            IdentifyArraysAccessedAsynchronously();

            DuplicateBarriers();

            if (GPUVerifyVCGenCommandLineOptions.IdentifySafeBarriers)
                IdentifySafeBarriers();

            if (!ProgramUsesBarrierInvariants())
                GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks = false;

            if (GPUVerifyVCGenCommandLineOptions.ArrayBoundsChecking)
                PerformArrayBoundsChecking();

            if (GPUVerifyVCGenCommandLineOptions.RemovePrivateArrayAccesses)
                EliminateLiteralIndexedPrivateArrays();

            if (GPUVerifyVCGenCommandLineOptions.RefinedAtomics)
                RefineAtomicAbstraction();

            var nonUniformVars = new List<Variable> { IdX, IdY, IdZ };
            if (!GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
                nonUniformVars.AddRange(new Variable[] { groupIdX, groupIdY, groupIdZ });

            UniformityAnalyser = DoUniformityAnalysis(nonUniformVars);

            if (GPUVerifyVCGenCommandLineOptions.ShowUniformityAnalysis)
                UniformityAnalyser.dump();

            // We now do modset analysis here because the previous passes add new
            // global variables, and the following two passes depend on the modset
            new ModSetCollector().DoModSetAnalysis(Program);

            DoVariableDefinitionAnalysis();

            DoReducedStrengthAnalysis();

            DoMayBePowerOfTwoAnalysis();

            DoArrayControlFlowAnalysis();

            DoCallSiteAnalysis();

            if (GPUVerifyVCGenCommandLineOptions.Inference)
            {
                foreach (var impl in Program.Implementations.ToList())
                {
                    if (!GPUVerifyVCGenCommandLineOptions.DisableInessentialLoopDetection)
                        LoopInvariantGenerator.EstablishDisabledLoops(this, impl);
                    LoopInvariantGenerator.PreInstrument(this, impl);
                }

                if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                    EmitProgram(outputFilename + "_pre_inference");
            }

            ConstantWriteInstrumenter.AddConstantWriteInstrumentation();

            if (GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams.Count > 0)
                AddParamsAsPreconditions();

            RaceInstrumenter.AddRaceCheckingInstrumentation();

            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
                NoAccessInstrumenter.AddNoAccessInstrumentation();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_instrumented");

            AbstractSharedState();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_abstracted");

            MergeBlocksIntoPredecessors();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_merged_pre_predication");

            if (GPUVerifyVCGenCommandLineOptions.WarpSync)
                AddWarpSyncs();

            MakeKernelPredicated();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_predicated");

            MergeBlocksIntoPredecessors(false);

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_merged_post_predication");

            MakeKernelDualised();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_dualised");

            if (GPUVerifyVCGenCommandLineOptions.NonDeterminiseUninterpretedFunctions)
            {
                NonDeterminiseUninterpretedFunctions();
                if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                    EmitProgram(outputFilename + "_ufs_removed");
            }

            RaceInstrumenter.AddRaceCheckingDeclarations();

            foreach (var b in barrierProcedures)
                GenerateBarrierImplementation(b);

            // We now do modset analysis here because the previous passes add new
            // global variables
            new ModSetCollector().DoModSetAnalysis(Program);

            if (GPUVerifyVCGenCommandLineOptions.OptimiseBarrierIntervals)
                OptimiseBarrierIntervals();

            GenerateStandardKernelContract();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
                EmitProgram(outputFilename + "_ready_to_verify");

            if (GPUVerifyVCGenCommandLineOptions.Inference)
            {
                ComputeInvariant();

                if (GPUVerifyVCGenCommandLineOptions.AbstractHoudini)
                    new AbstractHoudiniTransformation(this).DoAbstractHoudiniTransform();
            }

            // TODO: we do this before adding warp syncs because warp syncs
            // are added using structured commands, and the loop analysis involved
            // in adding capture states to loop heads demands an unstructured
            // representation.  It would be nicer to eliminate this ordering
            // constraint.
            AddLoopInvariantDisabledTags();
            AddCaptureStates();

            if (GPUVerifyVCGenCommandLineOptions.WarpSync)
            {
                GenerateWarpSyncs();
            }

            EmitProgram(outputFilename);
        }

        private void RemoveAtomicCalls()
        {
            throw new NotImplementedException();
        }

        private void PropagateProcedureWideInvariants()
        {
            foreach (var impl in Program.Implementations.ToList())
            {
                foreach (var inv in impl.Proc.Requires.Where(item => QKeyValue.FindBoolAttribute(item.Attributes, "procedure_wide_invariant")))
                {
                    foreach (var region in RootRegion(impl).SubRegions())
                    {
                        if (QKeyValue.FindBoolAttribute(inv.Attributes, "candidate"))
                            AddCandidateInvariant(region, inv.Condition, "procedure_wide", (QKeyValue)inv.Attributes.Clone());
                        else
                            region.AddInvariant(new AssertCmd(Token.NoToken, inv.Condition, (QKeyValue)inv.Attributes.Clone()));
                    }
                }

                // Remove pre-conditions representing procedure-wide invariants now
                // they have bee propagated.
                impl.Proc.Requires = impl.Proc.Requires.Where(item => !QKeyValue.FindBoolAttribute(item.Attributes, "procedure_wide_invariant")).ToList();
            }
        }

        private void RemoveUnnecessaryBlockSourceLocations()
        {
            // This is a work-around for the fact that Boogie sometimes
            // yields stack overflows (esp. under Windows).  To shorten
            // the Boogie program, we remove source location information
            // for non-loop head blocks.  These are not required, and
            // can end up bloating the program.
            // The longer-term solution is to avoid the stack overflows
            // in Boogie
            foreach (var impl in Program.Implementations)
            {
                var blockGraph = Program.ProcessLoops(impl);
                foreach (var b in impl.Blocks.Where(item => !blockGraph.Headers.Contains(item)))
                {
                    List<Cmd> newCmds = new List<Cmd>();
                    foreach (var c in b.Cmds)
                    {
                        var ac = c as AssertCmd;
                        if (ac != null && QKeyValue.FindBoolAttribute(ac.Attributes, "block_sourceloc"))
                            continue;

                        newCmds.Add(c);
                    }

                    b.Cmds = newCmds;
                }
            }
        }

        private void IdentifyArraysAccessedAsynchronously()
        {
            foreach (var asyncCall in Program.Blocks().Select(item => item.Cmds).SelectMany(item => item).
              OfType<CallCmd>().Where(item => QKeyValue.FindBoolAttribute(
                item.Attributes, "async_work_group_copy")))
            {
                Variable dstArray =
                  (asyncCall.Outs[1] as IdentifierExpr).Decl;
                Variable srcArray =
                  (asyncCall.Ins[1] as IdentifierExpr).Decl;
                Debug.Assert(KernelArrayInfo.GetGlobalAndGroupSharedArrays(true).Contains(dstArray));
                Debug.Assert(KernelArrayInfo.GetGlobalAndGroupSharedArrays(true).Contains(srcArray));
                ArraysAccessedByAsyncWorkGroupCopy[AccessType.WRITE].Add(dstArray.Name);
                ArraysAccessedByAsyncWorkGroupCopy[AccessType.READ].Add(srcArray.Name);
            }
        }

        private int DeepestSubregion(IRegion r)
        {
            if (r.SubRegions().Where(item => item.Identifier() != r.Identifier()).Count() == 0)
                return 0;

            return 1 + r.SubRegions().Where(item => item.Identifier() != r.Identifier()).Select(item => DeepestSubregion(item)).Max();
        }

        private void PrintLoopStatistics()
        {
            // For each implementation, dump the number of loops and the depth of the deepest loop nest to a file
            var loopsOutputFile = Path.GetFileNameWithoutExtension(GPUVerifyVCGenCommandLineOptions.InputFiles[0]) + ".loops";
            using (TokenTextWriter writer = new TokenTextWriter(loopsOutputFile, false))
            {
                foreach (var impl in Program.Implementations)
                {
                    writer.WriteLine("Implementation: " + impl.Name);
                    writer.WriteLine("Number of loops: " + RootRegion(impl).SubRegions().Count());
                    writer.WriteLine("Depth of deepest loop nest: " + DeepestSubregion(RootRegion(impl)));
                    writer.WriteLine();
                }
            }
        }

        private void AddParamsAsPreconditions()
        {
            foreach (List<string> param_values in GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams)
            {
                string target_name = "$" + param_values[0];

                // Locate the kernel with the given name
                bool found_flag = false;
                Procedure proc = null;
                foreach (KeyValuePair<Procedure, Implementation> entry in KernelProcedures)
                {
                    if (target_name == entry.Key.Name)
                    {
                        found_flag = true;
                        proc = entry.Key;
                        break;
                    }
                }

                if (found_flag == false)
                {
                    Console.WriteLine("Error: Couldn't find kernel " + target_name + ".");
                    Environment.Exit(1);
                }

                // Fail if too many params given (note that first
                // element of param_values is the name of the kernel)
                if (param_values.Count - 1 > proc.InParams.Count)
                {
                    Console.WriteLine("Error: Too many parameter values.");
                    Environment.Exit(1);
                }

                // Create requires clauses
                for (int ctr = 1; ctr < param_values.Count; ctr++)
                {
                    Variable v = proc.InParams[ctr - 1];
                    Expr v_expr = new IdentifierExpr(v.tok, v);
                    string val = param_values[ctr];

                    // Asterisk used to signify arbitrary value,
                    // hence no requires clause is needed.
                    if (val == "*")
                        continue;

                    if (!val.StartsWith("0x"))
                    {
                        Console.WriteLine("Error: Invalid hex string");
                        Environment.Exit(1);
                    }

                    val = val.Substring(2);
                    BigInteger arg = BigInteger.Parse(val, NumberStyles.HexNumber);

                    Expr val_expr = IntRep.GetLiteral(arg, v.TypedIdent.Type);
                    Expr v_eq_val = Expr.Eq(v_expr, val_expr);
                    proc.Requires.Add(new Requires(false, v_eq_val));
                }
            }
        }

        private void IdentifySafeBarriers()
        {
            var uni = DoUniformityAnalysis(new List<Variable> { IdX, IdY, IdZ });
            foreach (var b in barrierProcedures)
            {
                if (uni.IsUniform(b.Name))
                {
                    b.AddAttribute("safe_barrier", new object[] { });
                }
            }
        }

        private void PerformArrayBoundsChecking()
        {
            var boundsChecker = new ArrayBoundsChecker(this, Program);
            boundsChecker.CheckBounds(Program);
        }

        private void DuplicateBarriers()
        {
            // Make a separate barrier procedure for every barrier call.
            // This paves the way for barrier divergence optimisations
            // for specific barriers
            Contract.Requires(barrierProcedures.Count() == 1);
            Program.RemoveTopLevelDeclarations(x => x == barrierProcedures.First());
            barrierProcedures = new HashSet<Procedure>();
            int barrierCounter = 0;

            foreach (Block b in Program.Blocks().ToList())
            {
                List<Cmd> newCmds = new List<Cmd>();
                foreach (Cmd c in b.Cmds)
                {
                    var call = c as CallCmd;
                    if (call == null || !IsBarrier(call.Proc))
                    {
                        newCmds.Add(c);
                        continue;
                    }

                    Procedure newBarrier = new Duplicator().VisitProcedure(call.Proc);
                    Debug.Assert(IsBarrier(newBarrier));
                    newBarrier.Name = newBarrier.Name + "_duplicated_" + barrierCounter;
                    barrierCounter++;
                    var newCall = new CallCmd(call.tok, newBarrier.Name, call.Ins, call.Outs, call.Attributes);
                    newCall.Proc = newBarrier;
                    newCmds.Add(newCall);
                    Program.AddTopLevelDeclaration(newBarrier);
                    barrierProcedures.Add(newBarrier);
                    resContext.AddProcedure(newBarrier);
                }

                b.Cmds = newCmds;
            }
        }

        private void NonDeterminiseUninterpretedFunctions()
        {
            var ufRemover = new UninterpretedFunctionRemover();
            ufRemover.Eliminate(Program);
        }

        private void EliminateLiteralIndexedPrivateArrays()
        {
            // If a program contains private arrays that are only ever indexed by
            // literals, these can be eliminated.  This reduces the extent to which
            // arrays are used in the generated .bpl program, which may benefit
            // constraint solving.
            var eliminator = new LiteralIndexedArrayEliminator(this);
            eliminator.Eliminate(Program);
        }

        private void AddCaptureStates()
        {
            AddCaptureStatesToLoops();
            AddCaptureStatesAfterProcedureCalls();
        }

        private void AddCaptureStatesAfterProcedureCalls()
        {
            int counter = 0;
            foreach (var b in Program.Blocks())
            {
                List<Cmd> newCmds = new List<Cmd>();
                foreach (var c in b.Cmds)
                {
                    newCmds.Add(c);
                    var call = c as CallCmd;
                    if (call != null && !ProcedureIsInlined(call.Proc))
                    {
                        var key = new QKeyValue(Token.NoToken, "procedureName", new List<object> { call.callee }, null);
                        key = new QKeyValue(Token.NoToken, "captureState", new List<object> { "call_return_state_" + counter }, key);
                        newCmds.Add(new AssumeCmd(Token.NoToken, Expr.True, key));
                    }
                }

                b.Cmds = newCmds;
            }
        }

        private void AddCaptureStatesToLoops()
        {
            // Add the ability to get the state right before entering each loop,
            // at the loop head itself, and right before taking a back-edge
            int loopCounter = 0;
            foreach (var impl in Program.Implementations)
            {
                var cfg = Program.GraphFromImpl(impl);
                cfg.ComputeLoops();
                foreach (var header in cfg.Headers)
                {
                    AddStateCaptureToLoopHead(loopCounter, header);
                    AppendStateCaptureToBlocks("loop_back_edge_state", loopCounter, cfg.BackEdgeNodes(header));
                    AppendStateCaptureToBlocks("loop_entry_state", loopCounter, LoopEntryEdgeNodes(cfg, header));
                    loopCounter++;
                }
            }
        }

        private static IEnumerable<Block> LoopEntryEdgeNodes(Microsoft.Boogie.GraphUtil.Graph<Block> cfg, Block header)
        {
            return cfg.Predecessors(header).Where(item => !cfg.BackEdgeNodes(header).Contains(item));
        }

        private void AppendStateCaptureToBlocks(string stateNamePrefix, int loopCounter, IEnumerable<Block> blocks)
        {
            int counter = 0;
            foreach (var n in blocks)
            {
                var key = new QKeyValue(Token.NoToken, "captureState", new List<object> { stateNamePrefix + "_" + loopCounter + "_" + counter }, null);
                n.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.True, key));
                counter++;
            }
        }

        private static void AddStateCaptureToLoopHead(int loopCounter, Block b)
        {
            List<Cmd> newCmds = new List<Cmd>();
            var key = new QKeyValue(Token.NoToken, "captureState", new List<object> { "loop_head_state_" + loopCounter }, null);
            newCmds.Add(new AssumeCmd(Token.NoToken, Expr.True, key));
            newCmds.AddRange(b.Cmds);
            b.Cmds = newCmds;
        }

        private void CheckUserSuppliedLoopInvariants()
        {
            foreach (var impl in Program.Implementations)
            {
                var blockGraph = Program.ProcessLoops(impl);
                foreach (var b in impl.Blocks)
                {
                    bool validPositionForInvariant = blockGraph.Headers.Contains(b);
                    foreach (var c in b.Cmds)
                    {
                        var pc = c as PredicateCmd;
                        if (pc != null)
                        {
                            if (QKeyValue.FindBoolAttribute(pc.Attributes, "originated_from_invariant")
                              && !validPositionForInvariant)
                            {
                                var sourceLoc = new SourceLocationInfo(pc.Attributes, GPUVerifyVCGenCommandLineOptions.InputFiles[0], pc.tok);

                                Console.Write("\n" + sourceLoc.Top() + ": ");
                                Console.WriteLine("user-specified invariant does not appear at loop head.");
                                Console.WriteLine("\nNote: a common cause of this is due to the use of short-circuit operations;");
                                Console.WriteLine("      these should not be used in invariants.");
                                Environment.Exit(1);
                            }
                        }
                        else
                        {
                            validPositionForInvariant = false;
                        }
                    }
                }
            }
        }

        private void OptimiseBarrierIntervals()
        {
            var barrierIntervalsAnalysis = new BarrierIntervalsAnalysis(this, BarrierStrength.GROUP_SHARED);
            barrierIntervalsAnalysis.Compute();
            barrierIntervalsAnalysis.RemoveRedundantReads();
        }

        private void DoMayBePowerOfTwoAnalysis()
        {
            MayBePowerOfTwoAnalyser = new MayBePowerOfTwoAnalyser(this);
            MayBePowerOfTwoAnalyser.Analyse();
            RelationalPowerOfTwoAnalyser = new RelationalPowerOfTwoAnalyser(this);
            RelationalPowerOfTwoAnalyser.Analyse();
        }

        private void DoCallSiteAnalysis()
        {
            callSiteAnalyser = new CallSiteAnalyser(this);
            callSiteAnalyser.Analyse();
        }

        private void DoArrayControlFlowAnalysis()
        {
            arrayControlFlowAnalyser = new ArrayControlFlowAnalyser(this);
            arrayControlFlowAnalyser.Analyse();
        }

        private UniformityAnalyser DoUniformityAnalysis(List<Variable> nonUniformVars)
        {
            var entryPoints = new HashSet<Implementation>();
            if (GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis)
            {
                foreach (Implementation i in KernelProcedures.Values)
                {
                    if (i != null)
                        entryPoints.Add(i);
                }
            }

            var result = new UniformityAnalyser(
                Program, GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis, entryPoints, nonUniformVars);
            result.Analyse();
            return result;
        }

        private void DoVariableDefinitionAnalysis()
        {
            VarDefAnalysesRegion = Program.Implementations
                .ToDictionary(i => i, i => VariableDefinitionAnalysisRegion.Analyse(i, this));
        }

        private void DoReducedStrengthAnalysis()
        {
            ReducedStrengthAnalysesRegion = Program.Implementations
                .ToDictionary(i => i, i => ReducedStrengthAnalysisRegion.Analyse(i, this));
        }

        private void EmitProgram(string filename)
        {
            Utilities.IO.EmitProgram(Program, filename);
        }

        private void ComputeInvariant()
        {
            foreach (var impl in Program.Implementations.ToList())
            {
                LoopInvariantGenerator.PostInstrument(this, impl);
                if (ProcedureIsInlined(impl.Proc) || KernelProcedures.ContainsKey(impl.Proc))
                    continue;

                AddCandidateRequires(impl.Proc);
                RaceInstrumenter.AddRaceCheckingCandidateRequires(impl.Proc);
                AddCandidateEnsures(impl.Proc);
                RaceInstrumenter.AddRaceCheckingCandidateEnsures(impl.Proc);
            }
        }

        private void AddCandidateEnsures(Procedure proc)
        {
            HashSet<string> names = new HashSet<string>();
            foreach (Variable v in proc.OutParams)
                names.Add(Utilities.StripThreadIdentifier(v.Name));

            foreach (string name in names)
            {
                if (!UniformityAnalyser.IsUniform(proc.Name, name))
                    AddEqualityCandidateEnsures(proc, new LocalVariable(proc.tok, new TypedIdent(proc.tok, name, Microsoft.Boogie.Type.Int)));
            }
        }

        private void AddCandidateRequires(Procedure proc)
        {
            HashSet<string> names = new HashSet<string>();
            foreach (Variable v in proc.InParams)
            {
                names.Add(Utilities.StripThreadIdentifier(v.Name));
            }

            foreach (string name in names)
            {
                if (IsPredicateOrTemp(name))
                {
                    Debug.Assert(name.Equals("_P"));
                    Debug.Assert(!UniformityAnalyser.IsUniform(proc.Name));
                    AddCandidateRequires(proc, Expr.Eq(
                        new IdentifierExpr(proc.tok, new LocalVariable(proc.tok, new TypedIdent(proc.tok, name + "$1", Microsoft.Boogie.Type.Bool))),
                        new IdentifierExpr(proc.tok, new LocalVariable(proc.tok, new TypedIdent(proc.tok, name + "$2", Microsoft.Boogie.Type.Bool)))));
                }
                else
                {
                    if (!UniformityAnalyser.IsUniform(proc.Name, name))
                    {
                        if (!UniformityAnalyser.IsUniform(proc.Name))
                        {
                            AddPredicatedEqualityCandidateRequires(proc, new LocalVariable(proc.tok, new TypedIdent(proc.tok, name, Microsoft.Boogie.Type.Int)));
                        }

                        AddEqualityCandidateRequires(proc, new LocalVariable(proc.tok, new TypedIdent(proc.tok, name, Microsoft.Boogie.Type.Int)));
                    }
                }
            }
        }

        private void AddPredicatedEqualityCandidateRequires(Procedure proc, Variable v)
        {
            AddCandidateRequires(
                proc,
                Expr.Imp(
                    Expr.And(
                        new IdentifierExpr(proc.tok, new LocalVariable(proc.tok, new TypedIdent(proc.tok, "_P$1", Microsoft.Boogie.Type.Bool))),
                        new IdentifierExpr(proc.tok, new LocalVariable(proc.tok, new TypedIdent(proc.tok, "_P$2", Microsoft.Boogie.Type.Bool)))),
                    Expr.Eq(
                        new IdentifierExpr(proc.tok, new VariableDualiser(1, this, proc.Name).VisitVariable(v.Clone() as Variable)),
                        new IdentifierExpr(proc.tok, new VariableDualiser(2, this, proc.Name).VisitVariable(v.Clone() as Variable)))));
        }

        private void AddEqualityCandidateRequires(Procedure proc, Variable v)
        {
            AddCandidateRequires(
                proc,
                Expr.Eq(
                    new IdentifierExpr(proc.tok, new VariableDualiser(1, this, proc.Name).VisitVariable(v.Clone() as Variable)),
                    new IdentifierExpr(proc.tok, new VariableDualiser(2, this, proc.Name).VisitVariable(v.Clone() as Variable))));
        }

        private void AddEqualityCandidateEnsures(Procedure proc, Variable v)
        {
            AddCandidateEnsures(
                proc,
                Expr.Eq(
                    new IdentifierExpr(proc.tok, new VariableDualiser(1, this, proc.Name).VisitVariable(v.Clone() as Variable)),
                    new IdentifierExpr(proc.tok, new VariableDualiser(2, this, proc.Name).VisitVariable(v.Clone() as Variable))));
        }

        public void AddCandidateRequires(Procedure proc, Expr e)
        {
            Constant existentialBooleanConstant = Program.MakeExistentialBoolean();
            IdentifierExpr existentialBoolean = new IdentifierExpr(proc.tok, existentialBooleanConstant);
            proc.Requires.Add(new Requires(false, Expr.Imp(existentialBoolean, e)));
        }

        public void AddCandidateEnsures(Procedure proc, Expr e)
        {
            Constant existentialBooleanConstant = Program.MakeExistentialBoolean();
            IdentifierExpr existentialBoolean = new IdentifierExpr(proc.tok, existentialBooleanConstant);
            proc.Ensures.Add(new Ensures(false, Expr.Imp(existentialBoolean, e)));
        }

        public bool ContainsNamedVariable(HashSet<Variable> variables, string name)
        {
            foreach (Variable v in variables)
            {
                if (Utilities.StripThreadIdentifier(v.Name) == name)
                    return true;
            }

            return false;
        }

        public static bool IsPredicate(string v)
        {
            if (v.Length < 2)
                return false;

            if (!v.Substring(0, 1).Equals("p"))
                return false;

            for (int i = 1; i < v.Length; i++)
            {
                if (!char.IsDigit(v.ToCharArray()[i]))
                    return false;
            }

            return true;
        }

        public static bool IsPredicateOrTemp(string lv)
        {
            // We should improve this by having a general convention
            // for names of temporary or predicate variables
            if (lv.Length >= 2)
            {
                if (lv.Substring(0, 1).Equals("p") || lv.Substring(0, 1).Equals("v"))
                {
                    for (int i = 1; i < lv.Length; i++)
                    {
                        if (!char.IsDigit(lv.ToCharArray()[i]))
                            return false;
                    }

                    return true;
                }
            }

            if (lv.Contains("_HAVOC_"))
                return true;

            return (lv.Length >= 2 && lv.Substring(0, 2).Equals("_P"))
                || (lv.Length > 3 && lv.Substring(0, 3).Equals("_LC"))
                || (lv.Length > 5 && lv.Substring(0, 5).Equals("_temp"));
        }

        public Microsoft.Boogie.Type GetTypeOfIdX()
        {
            Contract.Requires(IdX != null);
            return IdX.TypedIdent.Type;
        }

        public Microsoft.Boogie.Type GetTypeOfIdY()
        {
            Contract.Requires(IdY != null);
            return IdY.TypedIdent.Type;
        }

        public Microsoft.Boogie.Type GetTypeOfIdZ()
        {
            Contract.Requires(IdZ != null);
            return IdZ.TypedIdent.Type;
        }

        public Microsoft.Boogie.Type GetTypeOfId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X"))
                return GetTypeOfIdX();
            if (dimension.Equals("Y"))
                return GetTypeOfIdY();
            if (dimension.Equals("Z"))
                return GetTypeOfIdZ();
            Debug.Assert(false);
            return null;
        }

        public bool KernelHasIdX()
        {
            return IdX != null;
        }

        public bool KernelHasIdY()
        {
            return IdY != null;
        }

        public bool KernelHasIdZ()
        {
            return IdZ != null;
        }

        public bool KernelHasGroupIdX()
        {
            return groupIdX != null;
        }

        public bool KernelHasGroupIdY()
        {
            return groupIdY != null;
        }

        public bool KernelHasGroupIdZ()
        {
            return groupIdZ != null;
        }

        public bool KernelHasNumGroupsX()
        {
            return numGroupsX != null;
        }

        public bool KernelHasNumGroupsY()
        {
            return numGroupsY != null;
        }

        public bool KernelHasNumGroupsZ()
        {
            return numGroupsZ != null;
        }

        public bool KernelHasGroupSizeX()
        {
            return groupSizeX != null;
        }

        public bool KernelHasGroupSizeY()
        {
            return groupSizeY != null;
        }

        public bool KernelHasGroupSizeZ()
        {
            return groupSizeZ != null;
        }

        private void GenerateStandardKernelContract()
        {
            RaceInstrumenter.AddKernelPrecondition();
            RaceInstrumenter.AddDefaultLoopInvariants();
            RaceInstrumenter.AddDefaultContracts();

            GeneratePreconditionsForDimension("X");
            GeneratePreconditionsForDimension("Y");
            GeneratePreconditionsForDimension("Z");

            foreach (Procedure proc in Program.TopLevelDeclarations.OfType<Procedure>())
            {
                if (ProcedureIsInlined(proc) || ProcedureHasNoImplementation(proc))
                    continue;

                Expr distinctLocalIds =
                    Expr.Or(
                        Expr.Or(
                            Expr.Neq(
                                new IdentifierExpr(Token.NoToken, MakeThreadId("X", 1)),
                                new IdentifierExpr(Token.NoToken, MakeThreadId("X", 2))),
                            Expr.Neq(
                                new IdentifierExpr(Token.NoToken, MakeThreadId("Y", 1)),
                                new IdentifierExpr(Token.NoToken, MakeThreadId("Y", 2)))),
                        Expr.Neq(
                            new IdentifierExpr(Token.NoToken, MakeThreadId("Z", 1)),
                            new IdentifierExpr(Token.NoToken, MakeThreadId("Z", 2))));

                proc.Requires.Add(new Requires(false, Expr.Imp(ThreadsInSameGroup(), distinctLocalIds)));

                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
                {
                    proc.Requires.Add(new Requires(false, ThreadsInSameGroup()));
                }

                if (KernelProcedures.ContainsKey(proc))
                {
                    bool foundNonUniform = false;
                    int indexOfFirstNonUniformParameter;
                    for (indexOfFirstNonUniformParameter = 0; indexOfFirstNonUniformParameter < proc.InParams.Count(); indexOfFirstNonUniformParameter++)
                    {
                        if (!UniformityAnalyser.IsUniform(proc.Name, Utilities.StripThreadIdentifier(proc.InParams[indexOfFirstNonUniformParameter].Name)))
                        {
                            foundNonUniform = true;
                            break;
                        }
                    }

                    if (foundNonUniform)
                    {
                        // I have a feeling this will never be reachable
                        int numberOfNonUniformParameters = (proc.InParams.Count() - indexOfFirstNonUniformParameter) / 2;
                        for (int i = indexOfFirstNonUniformParameter; i < numberOfNonUniformParameters; i++)
                        {
                            proc.Requires.Add(new Requires(
                                false,
                                Expr.Eq(
                                    new IdentifierExpr(proc.InParams[i].tok, proc.InParams[i]),
                                    new IdentifierExpr(proc.InParams[i + numberOfNonUniformParameters].tok, proc.InParams[i + numberOfNonUniformParameters]))));
                        }
                    }
                }
            }
        }

        public bool ProcedureHasNoImplementation(Procedure proc)
        {
            return !Program.Implementations.Select(i => i.Name).Contains(proc.Name);
        }

        public bool ProcedureIsInlined(Procedure proc)
        {
            return QKeyValue.FindIntAttribute(proc.Attributes, "inline", -1) == 1;
        }

        public Expr ThreadsInSameWarp()
        {
            Expr warpsize = IntRep.GetLiteral(GPUVerifyVCGenCommandLineOptions.WarpSize, IdType);
            IEnumerable<Expr> tids = (new int[] { 1, 2 }).Select(x => FlattenedThreadId(x));
            Expr[] sides = tids.Select(x => IntRep.MakeDiv(x, warpsize)).ToArray();
            return Expr.Eq(sides[0], sides[1]);
        }

        public Expr ThreadsInSameGroup()
        {
            if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
            {
                return Expr.True;
            }

            return Expr.And(
                    Expr.And(
                        Expr.Eq(
                            new IdentifierExpr(Token.NoToken, MakeGroupId("X", 1)),
                            new IdentifierExpr(Token.NoToken, MakeGroupId("X", 2))),
                        Expr.Eq(
                            new IdentifierExpr(Token.NoToken, MakeGroupId("Y", 1)),
                            new IdentifierExpr(Token.NoToken, MakeGroupId("Y", 2)))),
                    Expr.Eq(
                        new IdentifierExpr(Token.NoToken, MakeGroupId("Z", 1)),
                        new IdentifierExpr(Token.NoToken, MakeGroupId("Z", 2))));
        }

        public static int GetThreadSuffix(string p)
        {
            return int.Parse(p.Substring(p.IndexOf("$") + 1, p.Length - (p.IndexOf("$") + 1)));
        }

        private void GeneratePreconditionsForDimension(string dimension)
        {
            foreach (Declaration decl in Program.TopLevelDeclarations.ToList())
            {
                Procedure proc = decl as Procedure;
                if (proc == null || ProcedureIsInlined(proc) || ProcedureHasNoImplementation(proc))
                    continue;

                Expr groupSizePositive = IntRep.MakeSgt(new IdentifierExpr(Token.NoToken, GetGroupSize(dimension)), IntRep.GetZero(IdType));
                Expr numGroupsPositive = IntRep.MakeSgt(new IdentifierExpr(Token.NoToken, GetNumGroups(dimension)), IntRep.GetZero(IdType));
                Expr groupIdNonNegative = IntRep.MakeSge(new IdentifierExpr(Token.NoToken, GetGroupId(dimension)), IntRep.GetZero(IdType));
                Expr groupIdLessThanNumGroups = IntRep.MakeSlt(new IdentifierExpr(Token.NoToken, GetGroupId(dimension)), new IdentifierExpr(Token.NoToken, GetNumGroups(dimension)));

                proc.Requires.Add(new Requires(false, groupSizePositive));
                proc.Requires.Add(new Requires(false, numGroupsPositive));

                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
                {
                    proc.Requires.Add(new Requires(false, groupIdNonNegative));
                    proc.Requires.Add(new Requires(false, groupIdLessThanNumGroups));
                }
                else
                {
                    proc.Requires.Add(new Requires(false, new VariableDualiser(1, this, null).VisitExpr(groupIdNonNegative)));
                    proc.Requires.Add(new Requires(false, new VariableDualiser(2, this, null).VisitExpr(groupIdNonNegative)));
                    proc.Requires.Add(new Requires(false, new VariableDualiser(1, this, null).VisitExpr(groupIdLessThanNumGroups)));
                    proc.Requires.Add(new Requires(false, new VariableDualiser(2, this, null).VisitExpr(groupIdLessThanNumGroups)));
                }

                Expr threadIdNonNegative = IntRep.MakeSge(
                    new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)),
                    IntRep.GetZero(IdType));
                Expr threadIdLessThanGroupSize = IntRep.MakeSlt(
                    new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)),
                    new IdentifierExpr(Token.NoToken, GetGroupSize(dimension)));

                proc.Requires.Add(new Requires(false, new VariableDualiser(1, this, null).VisitExpr(threadIdNonNegative)));
                proc.Requires.Add(new Requires(false, new VariableDualiser(2, this, null).VisitExpr(threadIdNonNegative)));
                proc.Requires.Add(new Requires(false, new VariableDualiser(1, this, null).VisitExpr(threadIdLessThanGroupSize)));
                proc.Requires.Add(new Requires(false, new VariableDualiser(2, this, null).VisitExpr(threadIdLessThanGroupSize)));
            }
        }

        public Function GetOrCreateBVFunction(string functionName, string smtName, Microsoft.Boogie.Type resultType, params Microsoft.Boogie.Type[] argTypes)
        {
            Function f = (Function)resContext.LookUpProcedure(functionName);
            if (f != null)
                return f;

            f = new Function(
                Token.NoToken,
                functionName,
                new List<Variable>(argTypes.Select(t => new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, t))).ToArray()),
                new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, resultType)));
            f.AddAttribute("bvbuiltin", smtName);
            Program.AddTopLevelDeclaration(f);
            resContext.AddProcedure(f);
            return f;
        }

        public Function GetOrCreateIntFunction(string functionName, BinaryOperator.Opcode infixOp, Microsoft.Boogie.Type resultType, Microsoft.Boogie.Type lhsType, Microsoft.Boogie.Type rhsType)
        {
            Function f = (Function)resContext.LookUpProcedure(functionName);
            if (f != null)
                return f;

            List<Variable> inParams = new List<Variable>();
            Variable lhs = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "x", lhsType));
            Variable rhs = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "y", rhsType));
            inParams.Add(lhs);
            inParams.Add(rhs);

            f = new Function(
                Token.NoToken, functionName, inParams, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, resultType)));
            f.AddAttribute("inline", Expr.True);
            f.Body = Expr.Binary(infixOp, new IdentifierExpr(Token.NoToken, lhs), new IdentifierExpr(Token.NoToken, rhs));

            Program.AddTopLevelDeclaration(f);
            resContext.AddProcedure(f);
            return f;
        }

        public Function GetOrCreateBinaryUF(string functionName, Microsoft.Boogie.Type resultType, Microsoft.Boogie.Type lhsType, Microsoft.Boogie.Type rhsType)
        {
            Function f = (Function)resContext.LookUpProcedure(functionName);
            if (f != null)
                return f;

            var args = new List<Variable>
            {
                new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, lhsType)),
                new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, rhsType))
            };
            f = new Function(
                Token.NoToken, functionName, args, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, resultType)));
            Program.AddTopLevelDeclaration(f);
            resContext.AddProcedure(f);
            return f;
        }

        public Function GetOrCreateUnaryUF(string functionName, Microsoft.Boogie.Type resultType, Microsoft.Boogie.Type argType)
        {
            Function f = (Function)resContext.LookUpProcedure(functionName);
            if (f != null)
                return f;

            var args = new List<Variable>
            {
                new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, argType))
            };
            f = new Function(
                Token.NoToken, functionName, args, new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, resultType)));
            Program.AddTopLevelDeclaration(f);
            resContext.AddProcedure(f);
            return f;
        }

        public Constant GetGroupSize(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X"))
                return groupSizeX;
            if (dimension.Equals("Y"))
                return groupSizeY;
            if (dimension.Equals("Z"))
                return groupSizeZ;
            Debug.Assert(false);
            return null;
        }

        public Constant GetNumGroups(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X"))
                return numGroupsX;
            if (dimension.Equals("Y"))
                return numGroupsY;
            if (dimension.Equals("Z"))
                return numGroupsZ;
            Debug.Assert(false);
            return null;
        }

        public Constant MakeThreadId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            string name = null;
            if (dimension.Equals("X"))
                name = IdX.Name;
            if (dimension.Equals("Y"))
                name = IdY.Name;
            if (dimension.Equals("Z"))
                name = IdZ.Name;
            Debug.Assert(name != null);
            return new Constant(Token.NoToken, new TypedIdent(Token.NoToken, name, GetTypeOfId(dimension)));
        }

        public Constant MakeThreadId(string dimension, int number)
        {
            Constant resultWithoutThreadId = MakeThreadId(dimension);
            return new Constant(
                Token.NoToken,
                new TypedIdent(Token.NoToken, resultWithoutThreadId.Name + "$" + number, GetTypeOfId(dimension)));
        }

        public Constant GetGroupId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X"))
                return groupIdX;
            if (dimension.Equals("Y"))
                return groupIdY;
            if (dimension.Equals("Z"))
                return groupIdZ;
            Debug.Assert(false);
            return null;
        }

        public Constant MakeGroupId(string dimension, int number)
        {
            Constant resultWithoutThreadId = GetGroupId(dimension);
            return new Constant(Token.NoToken, new TypedIdent(Token.NoToken, resultWithoutThreadId.Name + "$" + number, GetTypeOfId(dimension)));
        }

        private void GenerateBarrierImplementation(Procedure barrierProcedure)
        {
            List<BigBlock> bigblocks = new List<BigBlock>();
            BigBlock barrierEntryBlock = new BigBlock(Token.NoToken, "__BarrierImpl", new List<Cmd>(), null, null);
            bigblocks.Add(barrierEntryBlock);

            Expr p1 = null, p2 = null, localFence1 = null, localFence2 = null, globalFence1 = null, globalFence2 = null;

            if (UniformityAnalyser.IsUniform(barrierProcedure.Name))
            {
                p1 = Expr.True;
                p2 = Expr.True;
            }

            foreach (Formal f in barrierProcedure.InParams)
            {
                int thread;
                string name = Utilities.StripThreadIdentifier(f.Name, out thread);
                if (name.Equals(barrierProcedureLocalFenceArgName))
                {
                    if (UniformityAnalyser.IsUniform(barrierProcedure.Name, name))
                    {
                        localFence1 = MakeFenceExpr(f);
                        localFence2 = MakeFenceExpr(f);
                    }
                    else
                    {
                        if (thread == 1)
                        {
                            localFence1 = MakeFenceExpr(f);
                        }
                        else
                        {
                            Debug.Assert(thread == 2);
                            localFence2 = MakeFenceExpr(f);
                        }
                    }
                }
                else if (name.Equals(barrierProcedureGlobalFenceArgName))
                {
                    if (UniformityAnalyser.IsUniform(barrierProcedure.Name, name))
                    {
                        globalFence1 = MakeFenceExpr(f);
                        globalFence2 = MakeFenceExpr(f);
                    }
                    else
                    {
                        if (thread == 1)
                        {
                            globalFence1 = MakeFenceExpr(f);
                        }
                        else
                        {
                            Debug.Assert(thread == 2);
                            globalFence2 = MakeFenceExpr(f);
                        }
                    }
                }
                else
                {
                    Debug.Assert(name.Equals("_P"));
                    if (thread == 1)
                    {
                        p1 = new IdentifierExpr(Token.NoToken, f);
                    }
                    else
                    {
                        Debug.Assert(thread == 2);
                        p2 = new IdentifierExpr(Token.NoToken, f);
                    }
                }
            }

            Debug.Assert(p1 != null);
            Debug.Assert(p2 != null);
            Debug.Assert(localFence1 != null);
            Debug.Assert(localFence2 != null);
            Debug.Assert(globalFence1 != null);
            Debug.Assert(globalFence2 != null);

            if (!QKeyValue.FindBoolAttribute(barrierProcedure.Attributes, "safe_barrier"))
            {
                Expr divergenceCondition = Expr.Imp(ThreadsInSameGroup(), Expr.Eq(p1, p2));
                Requires nonDivergenceRequires = new Requires(false, divergenceCondition);
                nonDivergenceRequires.Attributes =
                    new QKeyValue(Token.NoToken, "barrier_divergence", new List<object>(new object[] { }), null);
                barrierProcedure.Requires.Add(nonDivergenceRequires);
            }

            if (!GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
            {
                List<BigBlock> returnbigblocks = new List<BigBlock>();
                returnbigblocks.Add(new BigBlock(Token.NoToken, "__Disabled", new List<Cmd>(), null, new ReturnCmd(Token.NoToken)));
                StmtList returnstatement = new StmtList(returnbigblocks, barrierProcedure.tok);

                Expr ifGuard = Expr.Or(Expr.And(Expr.Not(p1), Expr.Not(p2)), Expr.And(ThreadsInSameGroup(), Expr.Or(Expr.Not(p1), Expr.Not(p2))));
                barrierEntryBlock.ec = new IfCmd(Token.NoToken, ifGuard, returnstatement, null, null);
            }

            var sharedArrays = KernelArrayInfo.GetGroupSharedArrays(true);
            sharedArrays = sharedArrays.Where(x => !KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(x)).ToList();
            if (sharedArrays.ToList().Count > 0)
            {
                bigblocks.AddRange(
                      MakeResetBlocks(Expr.And(p1, localFence1), sharedArrays.Where(x => KernelArrayInfo.GetGroupSharedArrays(false).Contains(x))));

                // This could be relaxed to take into account whether the threads are in different
                // groups, but for now we keep it relatively simple
                Expr atLeastOneEnabledWithLocalFence =
                  Expr.Or(Expr.And(p1, localFence1), Expr.And(p2, localFence2));

                if (SomeArrayModelledNonAdversarially(sharedArrays))
                {
                    var noAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ?
                      sharedArrays.Select(x => FindOrCreateNotAccessedVariable(x.Name, (x.TypedIdent.Type as MapType).Arguments[0])) :
                      Enumerable.Empty<Variable>();
                    var havocVars = sharedArrays.Concat(noAccessVars).ToList();
                    var cmd = new IfCmd(
                        Token.NoToken, atLeastOneEnabledWithLocalFence, new StmtList(MakeHavocBlocks(havocVars), Token.NoToken), null, null);
                    bigblocks.Add(new BigBlock(Token.NoToken, null, new List<Cmd>(), cmd, null));
                }
            }

            var globalArrays = KernelArrayInfo.GetGlobalArrays(true);
            globalArrays = globalArrays.Where(x => !KernelArrayInfo.GetReadOnlyGlobalAndGroupSharedArrays(true).Contains(x)).ToList();
            if (globalArrays.ToList().Count > 0)
            {
                bigblocks.AddRange(
                      MakeResetBlocks(Expr.And(p1, globalFence1), globalArrays.Where(x => KernelArrayInfo.GetGlobalArrays(false).Contains(x))));

                Expr threadsInSameGroupBothEnabledAtLeastOneGlobalFence =
                  Expr.And(Expr.And(ThreadsInSameGroup(), Expr.And(p1, p2)), Expr.Or(globalFence1, globalFence2));

                if (SomeArrayModelledNonAdversarially(globalArrays))
                {
                    var noAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ?
                      globalArrays.Select(x => FindOrCreateNotAccessedVariable(x.Name, (x.TypedIdent.Type as MapType).Arguments[0])) :
                      Enumerable.Empty<Variable>();
                    var havocVars = globalArrays.Concat(noAccessVars).ToList();
                    var cmd = new IfCmd(
                        Token.NoToken, threadsInSameGroupBothEnabledAtLeastOneGlobalFence, new StmtList(MakeHavocBlocks(havocVars), Token.NoToken), null, null);
                    bigblocks.Add(new BigBlock(Token.NoToken, null, new List<Cmd>(), cmd, null));
                }
            }

            if (RaceInstrumentationUtil.RaceCheckingMethod != RaceCheckingMethod.ORIGINAL && !GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
            {
                var havocId = new List<IdentifierExpr>
                {
                    new IdentifierExpr(Token.NoToken, new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_TRACKING", Microsoft.Boogie.Type.Bool)))
                };
                var havoc = new HavocCmd(Token.NoToken, havocId);
                bigblocks.Add(
                    new BigBlock(Token.NoToken, null, new List<Cmd> { havoc }, null, null));
            }

            StmtList statements = new StmtList(bigblocks, barrierProcedure.tok);
            Implementation barrierImplementation = new Implementation(
                barrierProcedure.tok,
                barrierProcedure.Name,
                new List<TypeVariable>(),
                barrierProcedure.InParams,
                barrierProcedure.OutParams,
                new List<Variable>(),
                statements);

            barrierImplementation.Resolve(resContext);

            AddInlineAttribute(barrierImplementation);
            AddInlineAttribute(barrierProcedure);

            barrierImplementation.Proc = barrierProcedure;

            Program.AddTopLevelDeclaration(barrierImplementation);
        }

        private NAryExpr MakeFenceExpr(Variable v)
        {
            return Expr.Neq(
                new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, v.TypedIdent)),
                IntRep.GetZero(IntRep.GetIntType(1)));
        }

        private List<BigBlock> MakeResetBlocks(Expr resetCondition, IEnumerable<Variable> variables)
        {
            Debug.Assert(variables.ToList().Count > 0);
            List<BigBlock> result = new List<BigBlock>();
            foreach (Variable v in variables)
            {
                result.Add(RaceInstrumenter.MakeResetReadWriteSetStatements(v, resetCondition));
            }

            Debug.Assert(result.Count > 0);
            return result;
        }

        private List<BigBlock> MakeHavocBlocks(ICollection<Variable> variables)
        {
            Debug.Assert(variables.Count > 0);
            List<BigBlock> result = new List<BigBlock>();
            foreach (Variable v in variables)
            {
                // Revisit: how to havoc NOT_ACCESSED vars properly
                if (!ArrayModelledAdversarially(v) || v.Name.Contains("_NOT_ACCESSED_"))
                {
                    result.Add(HavocSharedArray(v));
                }
            }

            Debug.Assert(result.Count > 0);
            return result;
        }

        private bool SomeArrayModelledNonAdversarially(IEnumerable<Variable> variables)
        {
            return variables.Any(v => !ArrayModelledAdversarially(v));
        }

        public static bool HasZDimension(Variable v)
        {
            if (v.TypedIdent.Type is MapType)
            {
                MapType mt = v.TypedIdent.Type as MapType;

                if (mt.Result is MapType)
                {
                    MapType mt2 = mt.Result as MapType;
                    if (mt2.Result is MapType)
                    {
                        Debug.Assert(!((mt2.Result as MapType).Result is MapType));
                        return true;
                    }
                }
            }

            return false;
        }

        private BigBlock HavocSharedArray(Variable v)
        {
            var havoc = new HavocCmd(Token.NoToken, new List<IdentifierExpr> { new IdentifierExpr(Token.NoToken, v) });
            return new BigBlock(Token.NoToken, null, new List<Cmd> { havoc }, null, null);
        }

        public static bool ModifiesSetContains(List<IdentifierExpr> modifies, IdentifierExpr v)
        {
            return modifies.Where(item => item.Name.Equals(v.Name)).Count() > 0;
        }

        private void AbstractSharedState()
        {
            new AdversarialAbstraction(this).Abstract();
        }

        public static string MakeBenignFlagVariableName(string name)
        {
            return "_WRITE_READ_BENIGN_FLAG_" + name;
        }

        public static GlobalVariable MakeBenignFlagVariable(string name)
        {
            return new GlobalVariable(
                Token.NoToken,
                new TypedIdent(Token.NoToken, MakeBenignFlagVariableName(name), Microsoft.Boogie.Type.Bool));
        }

        public GlobalVariable FindOrCreateArrayOffsetVariable(string varName)
        {
            string name = MakeArrayOffsetVariableName(varName);
            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (decl is GlobalVariable && ((GlobalVariable)decl).Name.Equals(name))
                {
                    return decl as GlobalVariable;
                }
            }

            GlobalVariable result = MakeArrayOffsetVariable(varName);

            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public GlobalVariable FindOrCreateNotAccessedVariable(string varName, Microsoft.Boogie.Type dtype)
        {
            string name = MakeNotAccessedVariableName(varName);
            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (decl is GlobalVariable && ((GlobalVariable)decl).Name.Equals(name))
                    return decl as GlobalVariable;
            }

            GlobalVariable result = MakeNotAccessedVariable(varName, dtype);

            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public GlobalVariable FindOrCreateAccessHasOccurredVariable(string varName, AccessType access)
        {
            foreach (var g in Program.TopLevelDeclarations.OfType<GlobalVariable>())
            {
                if (g.Name.Equals(RaceInstrumentationUtil.MakeHasOccurredVariableName(varName, access)))
                    return g;
            }

            GlobalVariable result = MakeAccessHasOccurredVariable(varName, access);
            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public Variable FindOrCreateAccessHasOccurredGhostVariable(string varName, string suffix, AccessType access, Implementation impl)
        {
            var ghostName = RaceInstrumentationUtil.MakeHasOccurredVariableName(varName, access) + "$ghost$" + suffix;
            foreach (var l in impl.LocVars)
            {
                if (l.Name.Equals(ghostName))
                    return l;
            }

            LocalVariable result = new LocalVariable(
                Token.NoToken, new TypedIdent(Token.NoToken, ghostName, Microsoft.Boogie.Type.Bool));
            impl.LocVars.Add(result);
            return result;
        }

        public Variable FindOrCreateOffsetVariable(string varName, AccessType access)
        {
            foreach (var g in Program.TopLevelDeclarations.OfType<Variable>())
            {
                if (g.Name.Equals(RaceInstrumentationUtil.MakeOffsetVariableName(varName, access)))
                    return g;
            }

            Variable result = RaceInstrumentationUtil.MakeOffsetVariable(varName, access, SizeTType);
            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public Variable FindOrCreateOffsetGhostVariable(string varName, string suffix, AccessType access, Implementation impl)
        {
            var ghostName = RaceInstrumentationUtil.MakeOffsetVariableName(varName, access) + "$ghost$" + suffix;
            foreach (var l in impl.LocVars)
            {
                if (l.Name.Equals(ghostName))
                    return l;
            }

            LocalVariable result = new LocalVariable(
                Token.NoToken, new TypedIdent(Token.NoToken, ghostName, SizeTType));
            impl.LocVars.Add(result);
            return result;
        }

        public Variable FindOrCreateValueVariable(
            string varName, AccessType access, Microsoft.Boogie.Type type)
        {
            foreach (var g in Program.TopLevelDeclarations.OfType<Variable>())
            {
                if (g.Name.Equals(RaceInstrumentationUtil.MakeValueVariableName(varName, access)))
                    return g;
            }

            Variable result = RaceInstrumentationUtil.MakeValueVariable(varName, access, type);
            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public GlobalVariable FindOrCreateBenignFlagVariable(string varName)
        {
            foreach (var g in Program.TopLevelDeclarations.OfType<GlobalVariable>())
            {
                if (g.Name.Equals(MakeBenignFlagVariableName(varName)))
                {
                    return g;
                }
            }

            GlobalVariable result = MakeBenignFlagVariable(varName);
            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public Variable FindOrCreateAsyncHandleVariable(string varName, AccessType access)
        {
            foreach (var g in Program.TopLevelDeclarations.OfType<Variable>())
            {
                if (g.Name.Equals(RaceInstrumentationUtil.MakeAsyncHandleVariableName(varName, access)))
                {
                    return g;
                }
            }

            Variable result = RaceInstrumentationUtil.MakeAsyncHandleVariable(varName, access, SizeTType);
            Program.AddTopLevelDeclaration(result);
            return result;
        }

        public GlobalVariable MakeArrayOffsetVariable(string varName)
        {
            return new GlobalVariable(
                Token.NoToken,
                new TypedIdent(Token.NoToken, MakeArrayOffsetVariableName(varName), SizeTType));
        }

        public static string MakeArrayOffsetVariableName(string varName)
        {
            return "_ARRAY_OFFSET_" + varName;
        }

        public static GlobalVariable MakeNotAccessedVariable(string varName, Microsoft.Boogie.Type dtype)
        {
            var v = new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, MakeNotAccessedVariableName(varName), dtype));
            v.Attributes = new QKeyValue(Token.NoToken, "check_access", new List<object>(new object[] { }), null);
            return v;
        }

        public static string MakeNotAccessedVariableName(string varName)
        {
            return "_NOT_ACCESSED_" + varName;
        }

        public static GlobalVariable MakeAccessHasOccurredVariable(string varName, AccessType access)
        {
            return new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, RaceInstrumentationUtil.MakeHasOccurredVariableName(varName, access), Microsoft.Boogie.Type.Bool));
        }

        public static IdentifierExpr MakeAccessHasOccurredExpr(string varName, AccessType access)
        {
            return new IdentifierExpr(Token.NoToken, MakeAccessHasOccurredVariable(varName, access));
        }

        public Function FindOrCreateOther(Microsoft.Boogie.Type type)
        {
            var otherFunctionName = type.IsBv ? "__other_bv" + type.BvBits : "__other_bv32";
            Function other = (Function)resContext.LookUpProcedure(otherFunctionName);
            if (other == null)
            {
                List<Variable> args = new List<Variable>();
                args.Add(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, type)));
                other = new Function(
                    Token.NoToken,
                    otherFunctionName,
                    args,
                    new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, string.Empty, type)));
            }

            return other;
        }

        private void MakeKernelDualised()
        {
            new KernelDualiser(this).DualiseKernel();
        }

        private void MakeKernelPredicated()
        {
            SmartBlockPredicator.Predicate(Program, proc => true, UniformityAnalyser);
        }

        private int Check()
        {
            var barrierProcedure = FindOrCreateBarrierProcedure();

            barrierProcedures.Add(barrierProcedure);

            if (ErrorCount > 0)
                return ErrorCount;

            if (barrierProcedure.InParams.Count() != 2)
                Error(barrierProcedure, "Barrier procedure must take exactly two arguments");
            else if (!barrierProcedure.InParams[0].TypedIdent.Type.Equals(IntRep.GetIntType(1)))
                Error(barrierProcedure, "First argument to barrier procedure must have type bv1");
            else if (!barrierProcedure.InParams[1].TypedIdent.Type.Equals(IntRep.GetIntType(1)))
                Error(barrierProcedure, "Second argument to barrier procedure must have type bv1");

            if (barrierProcedure.OutParams.Count() != 0)
                Error(barrierProcedure, "Barrier procedure must not return any results");

            if (barrierProcedure.InParams.Count() == 2)
            {
                barrierProcedureLocalFenceArgName = barrierProcedure.InParams[0].Name;
                barrierProcedureGlobalFenceArgName = barrierProcedure.InParams[1].Name;
            }

            KernelProcedures = GetKernelProcedures();

            if (!FindNonLocalVariables())
                return ErrorCount;

            if (GPUVerifyVCGenCommandLineOptions.ArraysToCheck != null)
            {
                // If the user provided arrays to which checking should be restricted, make sure
                // these really exist
                foreach (var v in GPUVerifyVCGenCommandLineOptions.ArraysToCheck)
                {
                    var keys = GlobalArraySourceNames.Where(x => x.Value == v).Select(x => x.Key);

                    bool reportError = !keys.Any();

                    if (!reportError)
                    {
                        foreach (var a in keys)
                        {
                            reportError = !KernelArrayInfo.GetGlobalAndGroupSharedArrays(true).Select(item => item.Name).Contains(a);

                            if (reportError)
                                break;
                        }
                    }

                    if (reportError)
                        Error(Token.NoToken, "Array name '" + v + "' specified for restricted checking is not found");
                }
            }

            return ErrorCount;
        }

        public bool IsThreadLocalIdConstant(Variable variable)
        {
            return variable.Name.Equals(IdX.Name) || variable.Name.Equals(IdY.Name) || variable.Name.Equals(IdZ.Name);
        }

        public bool IsGroupIdConstant(Variable variable)
        {
            return variable.Name.Equals(groupIdX.Name) || variable.Name.Equals(groupIdY.Name) || variable.Name.Equals(groupIdZ.Name);
        }

        public bool IsDualisedGroupIdConstant(Variable variable)
        {
            var name = Utilities.StripThreadIdentifier(variable.Name);
            return name.Equals(groupIdX.Name) || name.Equals(groupIdY.Name) || name.Equals(groupIdZ.Name);
        }

        public void AddCandidateInvariant(IRegion region, Expr e, string tag, string attribute)
        {
            AddCandidateInvariant(region, e, tag, new QKeyValue(Token.NoToken, attribute, new List<object>(), null));
        }

        public void AddCandidateInvariant(IRegion region, Expr e, string tag, QKeyValue attributes = null)
        {
            if (GPUVerifyVCGenCommandLineOptions.DoNotGenerateCandidates.Contains(tag))
            {
                return; // candidate *not* generated
            }

            PredicateCmd predicate = Program.CreateCandidateInvariant(e, tag);

            if (attributes != null)
            {
                attributes.AddLast(predicate.Attributes);
                predicate.Attributes = attributes;
            }

            region.AddInvariant(predicate);
        }

        public Implementation GetImplementation(string procedureName)
        {
            var relevant = Program.Implementations.Where(item => item.Name == procedureName);
            return relevant.Count() == 0 ? null : relevant.First();
        }

        public Procedure GetProcedure(string procedureName)
        {
            var relevant = Program.TopLevelDeclarations.OfType<Procedure>().Where(item => item.Name == procedureName);
            Debug.Assert(relevant.Count() > 0);
            return relevant.First();
        }

        public bool ContainsBarrierCall(IRegion loop)
        {
            return loop.Cmds().OfType<CallCmd>().Where(item => IsBarrier(item.Proc)).Count() > 0;
        }

        public bool ContainsUnsafeBarrierCall(IRegion loop)
        {
            return loop.Cmds().OfType<CallCmd>().Where(item => IsBarrier(item.Proc)
              && !QKeyValue.FindBoolAttribute(item.Proc.Attributes, "safe_barrier")).Count() > 0;
        }

        public static bool IsBarrier(Procedure proc)
        {
            return QKeyValue.FindBoolAttribute(proc.Attributes, "barrier");
        }

        public bool ArrayModelledAdversarially(Variable v)
        {
            if (GPUVerifyVCGenCommandLineOptions.AdversarialAbstraction)
                return true;

            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
                return false;

            if (KernelArrayInfo.GetAtomicallyAccessedArrays(true).Contains(v))
                return true;

            return !arrayControlFlowAnalyser.MayAffectControlFlow(v.Name);
        }

        public Expr GlobalIdExpr(string dimension)
        {
            var mul = IntRep.MakeMul(
                new IdentifierExpr(Token.NoToken, GetGroupId(dimension)),
                new IdentifierExpr(Token.NoToken, GetGroupSize(dimension)));
            return IntRep.MakeAdd(mul, new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)));
        }

        public IRegion RootRegion(Implementation impl)
        {
            return new UnstructuredRegion(Program, impl);
        }

        public static bool IsGivenConstant(Expr e, Constant c)
        {
            if (!(e is IdentifierExpr))
                return false;

            var varName = ((IdentifierExpr)e).Decl.Name;
            return Utilities.StripThreadIdentifier(varName) == Utilities.StripThreadIdentifier(c.Name);
        }

        public bool SubstIsGivenConstant(Implementation impl, Expr e, Constant c)
        {
            if (!(e is IdentifierExpr))
                return false;

            e = VarDefAnalysesRegion[impl].SubstDefinitions(e, impl.Name);
            return IsGivenConstant(e, c);
        }

        public Constant GetLocalIdConst(int dim)
        {
            switch (dim)
            {
                case 0:
                    return IdX;
                case 1:
                    return IdY;
                case 2:
                    return IdZ;
                default:
                    Debug.Assert(false);
                    return null;
            }
        }

        public Constant GetGroupIdConst(int dim)
        {
            switch (dim)
            {
                case 0:
                    return groupIdX;
                case 1:
                    return groupIdY;
                case 2:
                    return groupIdZ;
                default:
                    Debug.Assert(false);
                    return null;
            }
        }

        public Constant GetGroupSizeConst(int dim)
        {
            switch (dim)
            {
                case 0:
                    return groupSizeX;
                case 1:
                    return groupSizeY;
                case 2:
                    return groupSizeZ;
                default:
                    Debug.Assert(false);
                    return null;
            }
        }

        public bool IsLocalId(Expr e, int dim, Implementation impl)
        {
            return SubstIsGivenConstant(impl, e, GetLocalIdConst(dim));
        }

        public bool IsGlobalId(Expr e, int dim, Implementation impl)
        {
            e = VarDefAnalysesRegion[impl].SubstDefinitions(e, impl.Name);

            Expr lhs, rhs;
            if (IntRep.IsAdd(e, out lhs, out rhs) && e.Type.Equals(IdType))
            {
                Constant localId = GetLocalIdConst(dim);

                if (IsGivenConstant(rhs, localId))
                {
                    return IsGroupIdTimesGroupSize(lhs, dim);
                }

                if (IsGivenConstant(lhs, localId))
                {
                    return IsGroupIdTimesGroupSize(rhs, dim);
                }
            }

            return false;
        }

        private bool IsGroupIdTimesGroupSize(Expr expr, int dim)
        {
            Expr lhs, rhs;
            if (IntRep.IsMul(expr, out lhs, out rhs) && expr.Type.Equals(IdType))
            {
                if (IsGroupIdAndSize(dim, lhs, rhs))
                {
                    return true;
                }

                if (IsGroupIdAndSize(dim, rhs, lhs))
                {
                    return true;
                }
            }

            return false;
        }

        private bool IsGroupIdAndSize(int dim, Expr maybeGroupId, Expr maybeGroupSize)
        {
            return IsGivenConstant(maybeGroupId, GetGroupIdConst(dim))
                && IsGivenConstant(maybeGroupSize, GetGroupSizeConst(dim));
        }

        public Expr MaybeDualise(Expr e, int id, string procName)
        {
            if (id == 0 || e == null)
                return e;
            else
                return (Expr)new VariableDualiser(id, this, procName).Visit(e.Clone());
        }

        public static bool IsConstantInCurrentRegion(IdentifierExpr expr)
        {
            return (expr.Decl is Constant)
                || (expr.Decl is Formal && ((Formal)expr.Decl).InComing);
        }

        public Expr GroupSharedIndexingExpr(int thread)
        {
            if (thread == 1)
            {
                return new LiteralExpr(Token.NoToken, BigNum.FromInt(1), 1);
            }
            else
            {
                var args = new List<Expr>
                {
                    ThreadsInSameGroup(),
                    new LiteralExpr(Token.NoToken, BigNum.FromInt(1), 1),
                    new LiteralExpr(Token.NoToken, BigNum.FromInt(0), 1)
                };
                return new NAryExpr(Token.NoToken, new IfThenElse(Token.NoToken), args);
            }
        }

        public bool ProgramUsesBarrierInvariants()
        {
            foreach (var b in Program.Blocks())
            {
                foreach (Cmd c in b.Cmds)
                {
                    if (c is CallCmd)
                    {
                        if (QKeyValue.FindBoolAttribute((c as CallCmd).Proc.Attributes, "barrier_invariant"))
                        {
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        public static void AddInlineAttribute(Declaration d)
        {
            d.AddAttribute("inline", new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(1)) });
        }

        // This finds instances where the only atomic used on an array satisfy forall n,m f^n(x) != f^m(x)
        // Technically unsound due to machine integers, but unlikely in practice due to, e.g., the need to call atomic_inc >2^32 times
        private void RefineAtomicAbstraction()
        {
            // First, pass over the program looking for uses of atomics, recording (Array,Function) pairs
            Dictionary<Variable, HashSet<string>> funcs_used = new Dictionary<Variable, HashSet<string>>();
            Dictionary<Variable, HashSet<Expr>> args_used = new Dictionary<Variable, HashSet<Expr>>();
            foreach (Block b in Program.Blocks())
            {
                foreach (Cmd c in b.Cmds)
                {
                    if (c is CallCmd)
                    {
                        CallCmd call = c as CallCmd;
                        if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic"))
                        {
                            Variable v = (call.Ins[0] as IdentifierExpr).Decl;

                            if (funcs_used.ContainsKey(v))
                                funcs_used[v].Add(QKeyValue.FindStringAttribute(call.Attributes, "atomic_function"));
                            else
                                funcs_used.Add(v, new HashSet<string>(new string[] { QKeyValue.FindStringAttribute(call.Attributes, "atomic_function") }));

                            Expr arg = QKeyValue.FindExprAttribute(call.Attributes, "arg1");
                            if (arg != null)
                            {
                                if (args_used.ContainsKey(v))
                                    args_used[v].Add(arg);
                                else
                                    args_used.Add(v, new HashSet<Expr>(new Expr[] { arg }));
                            }
                        }
                    }
                }
            }

            // Then, for every array that only used a single monotonic atomic function, pass over the program again, logging offset constraints
            string[] monotonics = new string[]
            {
                "__bugle_atomic_inc", "__bugle_atomic_dec", "__bugle_atomic_add", "__bugle_atomic_sub", "__atomicAdd", "__atomicSub"
            };
            Expr variables = null;
            Expr offset = null;
            int parts = 0;
            foreach (KeyValuePair<Variable, HashSet<string>> pair in funcs_used)
            {
                // If it's a refinable function, and either: (a) has no arguments (is inc or dec), or (b) has 1 argument and that argument is a non-zero constant
                if (pair.Value.Count == 1
                    && monotonics.Any(x => pair.Value.First().StartsWith(x))
                        && (!args_used.ContainsKey(pair.Key)
                            || (args_used[pair.Key].Count == 1
                                && args_used[pair.Key].All(arg => (arg is LiteralExpr) && !arg.Equals(IntRep.GetZero(arg.Type))))))
                {
                    foreach (Block b in Program.Blocks())
                    {
                        List<Cmd> result = new List<Cmd>();
                        foreach (Cmd c in b.Cmds)
                        {
                            result.Add(c);
                            if (c is CallCmd)
                            {
                                CallCmd call = c as CallCmd;
                                if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic") && (call.Ins[0] as IdentifierExpr).Decl.Equals(pair.Key))
                                {
                                    if (variables == null)
                                    {
                                        variables = call.Outs[0];
                                        offset = call.Ins[1];
                                        parts = QKeyValue.FindIntAttribute(call.Attributes, "parts", 0);
                                    }
                                    else
                                    {
                                        variables = new BvConcatExpr(Token.NoToken, call.Outs[0], variables);
                                        variables.Type = variables.ShallowType;
                                    }

                                    if (QKeyValue.FindIntAttribute(call.Attributes, "part", -1) == parts)
                                    {
                                        AssumeCmd assume = new AssumeCmd(Token.NoToken, Expr.True);
                                        assume.Attributes = new QKeyValue(Token.NoToken, "atomic_refinement", new List<object>(new object[] { }), null);
                                        assume.Attributes = new QKeyValue(Token.NoToken, "variable", new List<object>(new object[] { variables }), assume.Attributes);
                                        assume.Attributes = new QKeyValue(Token.NoToken, "offset", new List<object>(new object[] { offset }), assume.Attributes);
                                        assume.Attributes = new QKeyValue(Token.NoToken, "arrayref", new List<object>(new object[] { pair.Key.Name }), assume.Attributes);
                                        result.Add(assume);
                                        variables = null;
                                    }
                                }
                            }
                        }

                        b.Cmds = result;
                    }
                }
            }
        }

        public GlobalVariable FindOrCreateUsedMap(string arrayName, Microsoft.Boogie.Type elementType)
        {
            string name = "_USED_" + arrayName;

            var candidateVariables = Program.TopLevelDeclarations.OfType<GlobalVariable>().Where(item => item.Name.Equals(name));
            if (candidateVariables.Count() > 0)
            {
                Debug.Assert(candidateVariables.Count() == 1);
                return candidateVariables.First();
            }

            var mapType = new MapType(
                Token.NoToken, new List<TypeVariable>(), new List<Microsoft.Boogie.Type> { elementType }, Microsoft.Boogie.Type.Bool);
            mapType = new MapType(
                Token.NoToken, new List<TypeVariable>(), new List<Microsoft.Boogie.Type> { SizeTType }, mapType);

            GlobalVariable usedMap = new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, name, mapType));
            usedMap.Attributes = new QKeyValue(Token.NoToken, "atomic_usedmap", new List<object>(), null);

            if (KernelArrayInfo.GetGlobalArrays(true).Any(item => item.Name.Equals(arrayName)))
            {
                usedMap.Attributes = new QKeyValue(Token.NoToken, "atomic_global", new List<object>(), usedMap.Attributes);
            }
            else if (KernelArrayInfo.GetGroupSharedArrays(true).Any(item => item.Name.Equals(arrayName)))
            {
                usedMap.Attributes = new QKeyValue(Token.NoToken, "atomic_group_shared", new List<object>(), usedMap.Attributes);
            }

            Program.AddTopLevelDeclaration(usedMap);
            return usedMap;
        }

        private Expr FlattenedThreadId(int thread)
        {
            return IntRep.MakeAdd(
                Expr.Ident(MakeThreadId("X", thread)),
                IntRep.MakeAdd(
                    IntRep.MakeMul(
                        Expr.Ident(MakeThreadId("Y", thread)),
                        Expr.Ident(GetGroupSize("X"))),
                    IntRep.MakeMul(
                        Expr.Ident(MakeThreadId("Z", thread)),
                        IntRep.MakeMul(Expr.Ident(GetGroupSize("X")), Expr.Ident(GetGroupSize("Y"))))));
        }

        private void AddWarpSyncs()
        {
            foreach (Declaration d in Program.TopLevelDeclarations)
            {
                if (d is Implementation)
                {
                    Implementation impl = d as Implementation;
                    impl.Blocks = impl.Blocks.Select(AddWarpSyncs).ToList();
                }
            }

            foreach (Procedure proto in warpSyncs.Values)
            {
                AddInlineAttribute(proto);
                Program.AddTopLevelDeclaration(proto);
            }
        }

        private Block AddWarpSyncs(Block b)
        {
            // TODO: this code is hacky and needs an overhaul
            var result = new List<Cmd>();
            foreach (Cmd c in b.Cmds)
            {
                if (c is CallCmd)
                {
                    CallCmd call = c as CallCmd;
                    if (call.callee.StartsWith("_LOG_"))
                    {
                        string array;
                        AccessType kind;
                        if (call.callee.StartsWith("_LOG_ATOMIC"))
                        {
                            kind = AccessType.ATOMIC;
                            array = call.callee.Substring(12); // "_LOG_ATOMIC_" is 14 characters
                        }
                        else if (call.callee.StartsWith("_LOG_READ"))
                        {
                            kind = AccessType.READ;
                            array = call.callee.Substring(10); // "_LOG_READ_" is 12 characters
                        }
                        else
                        {
                            Debug.Assert(call.callee.StartsWith("_LOG_WRITE"));
                            kind = AccessType.WRITE;
                            array = call.callee.Substring(11); // "_LOG_WRITE_" is 13 characters
                        }

                        // Manual resolving
                        Variable arrayVar = KernelArrayInfo.GetAllArrays(true).Where(v => v.Name.Equals(array)).First();
                        Procedure proto = FindOrCreateWarpSync(arrayVar, kind, true);
                        CallCmd wsCall = new CallCmd(Token.NoToken, proto.Name, new List<Expr>(), new List<IdentifierExpr>());
                        wsCall.Proc = proto;
                        result.Add(wsCall);
                    }
                }

                result.Add(c);
                if (c is CallCmd)
                {
                    CallCmd call = c as CallCmd;
                    if (call.callee.StartsWith("_CHECK_"))
                    {
                        string array;
                        AccessType kind;
                        if (call.callee.StartsWith("_CHECK_ATOMIC"))
                        {
                            kind = AccessType.ATOMIC;
                            array = call.callee.Substring(14); // "_CHECK_ATOMIC_" is 14 characters
                        }
                        else if (call.callee.StartsWith("_CHECK_READ"))
                        {
                            kind = AccessType.READ;
                            array = call.callee.Substring(12); // "_CHECK_READ_" is 12 characters
                        }
                        else
                        {
                            Debug.Assert(call.callee.StartsWith("_CHECK_WRITE"));
                            kind = AccessType.WRITE;
                            array = call.callee.Substring(13); // "_CHECK_WRITE_" is 13 characters
                        }

                        // Manual resolving
                        Variable arrayVar = KernelArrayInfo.GetAllArrays(true).Where(v => v.Name.Equals(array)).First();
                        Procedure proto = FindOrCreateWarpSync(arrayVar, kind, false);
                        CallCmd wsCall = new CallCmd(Token.NoToken, proto.Name, new List<Expr>(), new List<IdentifierExpr>());
                        wsCall.Proc = proto;
                        result.Add(wsCall);
                    }
                }
            }

            b.Cmds = result;
            return b;
        }

        private Procedure FindOrCreateWarpSync(Variable array, AccessType kind, bool pre)
        {
            Tuple<Variable, AccessType, bool> key = new Tuple<Variable, AccessType, bool>(array, kind, pre);
            if (!warpSyncs.ContainsKey(key))
            {
                Procedure proto = new Procedure(
                    Token.NoToken,
                    (pre ? "_PRE" : "_POST") + "_WARP_SYNC_" + array.Name + "_" + kind,
                    new List<TypeVariable>(),
                    new List<Variable>(),
                    new List<Variable>(),
                    new List<Requires>(),
                    new List<IdentifierExpr>(),
                    new List<Ensures>());
                warpSyncs[key] = proto;
            }

            return warpSyncs[key];
        }

        private void GenerateWarpSyncs()
        {
            foreach (Tuple<Variable, AccessType, bool> pair in warpSyncs.Keys)
            {
                Variable v = pair.Item1;
                AccessType kind = pair.Item2;
                bool pre = pair.Item3;
                Procedure syncProcedure = warpSyncs[pair];

                Expr p1 = null;
                Expr p2 = null;

                if (UniformityAnalyser.IsUniform(syncProcedure.Name))
                {
                    p1 = Expr.True;
                    p2 = Expr.True;
                }
                else
                {
                    // If not uniform, should be predicated -- we don't take any other parameters...
                    p1 = Expr.Ident(syncProcedure.InParams[0]);
                    p2 = Expr.Ident(syncProcedure.InParams[1]);
                }

                // Implementation
                List<Cmd> then = new List<Cmd>();
                if (pre)
                {
                    var reset_needs = new[]
                    {
                        new { Kind = AccessType.READ,   Resets = new[] { AccessType.WRITE, AccessType.ATOMIC } },
                        new { Kind = AccessType.WRITE,  Resets = new[] { AccessType.READ, AccessType.WRITE, AccessType.ATOMIC } },
                        new { Kind = AccessType.ATOMIC, Resets = new[] { AccessType.READ, AccessType.WRITE } }
                    };

                    foreach (AccessType a in reset_needs.Where(x => x.Kind == kind).First().Resets)
                    {
                        Variable accessVariable = FindOrCreateAccessHasOccurredVariable(v.Name, a);
                        then.Add(new AssumeCmd(Token.NoToken, Expr.Not(Expr.Ident(accessVariable))));
                    }
                }
                else
                {
                    Variable accessVariable = FindOrCreateAccessHasOccurredVariable(v.Name, kind);
                    then.Add(new AssumeCmd(Token.NoToken, Expr.Not(Expr.Ident(accessVariable))));
                }

                List<BigBlock> thenblocks = new List<BigBlock>();
                thenblocks.Add(new BigBlock(Token.NoToken, "reset_warps", then, null, null));

                if (kind == AccessType.WRITE && !ArrayModelledAdversarially(v))
                {
                    thenblocks.AddRange(MakeHavocBlocks(new Variable[] { v }));
                }

                Expr condition = Expr.And(Expr.Eq(p1, p2), Expr.And(ThreadsInSameGroup(), ThreadsInSameWarp()));

                IfCmd ifcmd = new IfCmd(Token.NoToken, condition, new StmtList(thenblocks, Token.NoToken), /* another IfCmd for elsif */ null, /* then branch */ null);

                List<BigBlock> blocks = new List<BigBlock>();
                blocks.Add(new BigBlock(Token.NoToken, "entry", new List<Cmd>(), ifcmd, null));

                Implementation method = new Implementation(
                    Token.NoToken,
                    syncProcedure.Name,
                    new List<TypeVariable>(),
                    syncProcedure.InParams,
                    new List<Variable>(),
                    new List<Variable>(),
                    new StmtList(blocks, Token.NoToken));

                AddInlineAttribute(method);
                Program.AddTopLevelDeclaration(method);
            }
        }

        public void AddRegionWithLoopInvariantsDisabled(IRegion region)
        {
            regionsWithLoopInvariantsDisabled.Add(region.Identifier());
        }

        public bool RegionHasLoopInvariantsDisabled(IRegion region)
        {
            return regionsWithLoopInvariantsDisabled.Contains(region.Identifier());
        }

        private void AddLoopInvariantDisabledTags()
        {
            foreach (var impl in Program.Implementations)
            {
                foreach (var region in RootRegion(impl).SubRegions())
                {
                    if (RegionHasLoopInvariantsDisabled(region))
                        region.AddLoopInvariantDisabledTag();
                }
            }
        }

        private bool TryGetArrayFromPrefixedString(string s, string prefix, out Variable v)
        {
            v = null;
            if (s.StartsWith(prefix))
            {
                foreach (var a in KernelArrayInfo.GetGlobalAndGroupSharedArrays(true))
                {
                    if (a.Name.Equals(s.Substring(prefix.Length)))
                    {
                        v = a;
                        return true;
                    }
                }
            }

            return false;
        }

        public bool TryGetArrayFromAccessHasOccurred(string s, AccessType access, out Variable v)
        {
            return TryGetArrayFromPrefixedString(s, "_" + access + "_HAS_OCCURRED_", out v);
        }

        private bool TryGetArrayFromLogOrCheckProcedure(string s, AccessType access, string logOrCheck, out Variable v)
        {
            return TryGetArrayFromPrefixedString(s, "_" + logOrCheck + "_" + access + "_", out v);
        }

        public bool TryGetArrayFromLogProcedure(string s, AccessType access, out Variable v)
        {
            return TryGetArrayFromLogOrCheckProcedure(s, access, "LOG", out v);
        }

        public bool TryGetArrayFromCheckProcedure(string s, AccessType access, out Variable v)
        {
            return TryGetArrayFromLogOrCheckProcedure(s, access, "CHECK", out v);
        }

        public Variable FindOrCreateEnabledVariable()
        {
            string enabledVariableName = "__enabled";
            Variable enabledVariable = (Variable)resContext.LookUpVariable(enabledVariableName);
            if (enabledVariable == null)
            {
                enabledVariable = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, enabledVariableName, Microsoft.Boogie.Type.Bool), false);
                enabledVariable.AddAttribute("__enabled");
                resContext.AddVariable(enabledVariable, true);
            }

            return enabledVariable;
        }

        public Expr FindOrCreateAsyncNoHandleConstant()
        {
            string name = "_ASYNC_NO_HANDLE";
            var candidates = Program.TopLevelDeclarations.OfType<Constant>().Where(item => item.Name == name);
            if (candidates.Count() > 0)
            {
                Debug.Assert(candidates.Count() == 1);
                return Expr.Ident(candidates.First());
            }

            Constant asyncNoHandleConstant = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, name, SizeTType), false);
            Axiom equalsZero = new Axiom(Token.NoToken, Expr.Eq(Expr.Ident(asyncNoHandleConstant), IntRep.GetZero(SizeTType)));
            Program.AddTopLevelDeclarations(new Declaration[] { asyncNoHandleConstant, equalsZero });
            return Expr.Ident(asyncNoHandleConstant);
        }

        public void StripOutAnnotationsForDisabledArrays()
        {
            // If the user has asked for checking of an array A to be disabled then we
            // (a) remove any "*_HAS_OCCURRED" variables relating to A, and (b)
            // replace uses of these variables with "false"
            foreach (var v in KernelArrayInfo.GetGlobalAndGroupSharedArrays(true))
            {
                if (KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(v))
                {
                    continue;
                }

                // (a) Eliminate the relevant "*_HAS_OCCURRED" variables, tracking which were eliminated
                var newDecls = new List<Declaration>();
                var variablesToEliminate = new HashSet<Variable>();
                foreach (var d in Program.TopLevelDeclarations)
                {
                    if (d is Variable && Utilities.IsAccessHasOccurredVariable((Variable)d, v.Name))
                    {
                        variablesToEliminate.Add((Variable)d);
                        continue;
                    }

                    newDecls.Add(d);
                }

                Program.TopLevelDeclarations = newDecls;

                // (b) Substitute all eliminated variables for "false"
                foreach (var b in Program.Blocks())
                {
                    b.cmds = b.cmds.Select(c => Substituter.Apply(
                      x => variablesToEliminate.Contains(x) ? (Expr)Expr.False : (Expr)Expr.Ident(x), c)).ToList();
                }

                ExpressionSimplifier.Simplify(Program, this.IntRep);
            }
        }
    }
}
