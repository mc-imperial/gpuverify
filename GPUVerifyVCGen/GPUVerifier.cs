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
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using System.Numerics;
using System.Globalization;
using Microsoft.Boogie;
using Microsoft.Boogie.Houdini;
using Microsoft.Basetypes;

namespace GPUVerify
{
    public class InferenceStages {
      public const int BASIC_CANDIDATE_STAGE = 0;
      internal static int NO_READ_WRITE_CANDIDATE_STAGE = 0;
      internal static int ACCESS_PATTERN_CANDIDATE_STAGE = 0;
    }

    internal class GPUVerifier : CheckingContext
    {
        public string outputFilename;
        public Program Program;
        public IntegerRepresentation IntRep;
        public ResolutionContext ResContext;

        public Dictionary<Procedure, Implementation> KernelProcedures;
        public Dictionary<string, string> GlobalArraySourceNames;

        private HashSet<Procedure> BarrierProcedures = new HashSet<Procedure>();
        private Dictionary<Tuple<Variable,AccessType,bool>,Procedure> WarpSyncs = new Dictionary<Tuple<Variable,AccessType,bool>,Procedure>();
        public string BarrierProcedureLocalFenceArgName;
        public string BarrierProcedureGlobalFenceArgName;

        private HashSet<object> RegionsWithLoopInvariantsDisabled = new HashSet<object>();

        internal IKernelArrayInfo KernelArrayInfo = new KernelArrayInfoLists();

        private HashSet<string> ReservedNames = new HashSet<string>();

        internal const string _SIZE_T_BITS_TYPE = "_SIZE_T_TYPE";
        public readonly int size_t_bits;
        public readonly int id_size_bits;

        internal HashSet<string> OnlyThread1 = new HashSet<string>();
        internal HashSet<string> OnlyThread2 = new HashSet<string>();

        internal const string LOCAL_ID_X_STRING = "local_id_x";
        internal const string LOCAL_ID_Y_STRING = "local_id_y";
        internal const string LOCAL_ID_Z_STRING = "local_id_z";

        internal static Constant _X = null;
        internal static Constant _Y = null;
        internal static Constant _Z = null;

        internal const string GROUP_SIZE_X_STRING = "group_size_x";
        internal const string GROUP_SIZE_Y_STRING = "group_size_y";
        internal const string GROUP_SIZE_Z_STRING = "group_size_z";

        internal static Constant _GROUP_SIZE_X = null;
        internal static Constant _GROUP_SIZE_Y = null;
        internal static Constant _GROUP_SIZE_Z = null;

        internal const string GROUP_ID_X_STRING = "group_id_x";
        internal const string GROUP_ID_Y_STRING = "group_id_y";
        internal const string GROUP_ID_Z_STRING = "group_id_z";

        internal static Constant _GROUP_X = null;
        internal static Constant _GROUP_Y = null;
        internal static Constant _GROUP_Z = null;

        internal const string NUM_GROUPS_X_STRING = "num_groups_x";
        internal const string NUM_GROUPS_Y_STRING = "num_groups_y";
        internal const string NUM_GROUPS_Z_STRING = "num_groups_z";

        internal static Constant _NUM_GROUPS_X = null;
        internal static Constant _NUM_GROUPS_Y = null;
        internal static Constant _NUM_GROUPS_Z = null;

        internal IRaceInstrumenter RaceInstrumenter;
        internal INoAccessInstrumenter NoAccessInstrumenter;
        internal IConstantWriteInstrumenter ConstantWriteInstrumenter;

        internal UniformityAnalyser uniformityAnalyser;
        internal MayBePowerOfTwoAnalyser mayBePowerOfTwoAnalyser;
        internal RelationalPowerOfTwoAnalyser relationalPowerOfTwoAnalyser;
        internal ArrayControlFlowAnalyser arrayControlFlowAnalyser;
        internal CallSiteAnalyser callSiteAnalyser;
        internal Dictionary<Implementation, VariableDefinitionAnalysis> varDefAnalyses;
        internal Dictionary<Implementation, ReducedStrengthAnalysis> reducedStrengthAnalyses;

        internal Dictionary<AccessType, HashSet<string>> ArraysAccessedByAsyncWorkGroupCopy;

        internal GPUVerifier(string filename, Program program, ResolutionContext rc)
            : base((IErrorSink)null)
        {
            this.outputFilename = filename;
            this.Program = program;
            this.ResContext = rc;
            this.IntRep = GPUVerifyVCGenCommandLineOptions.MathInt ?
                (IntegerRepresentation)new MathIntegerRepresentation(this) :
                (IntegerRepresentation)new BVIntegerRepresentation(this);

            this.size_t_bits = GetSizeTBits();
            this.id_size_bits = GetIdSizeBits();

            this.ArraysAccessedByAsyncWorkGroupCopy = new Dictionary<AccessType,HashSet<string>>();
            this.ArraysAccessedByAsyncWorkGroupCopy[AccessType.READ] = new HashSet<string>();
            this.ArraysAccessedByAsyncWorkGroupCopy[AccessType.WRITE] = new HashSet<string>();

            if (this.size_t_bits < this.id_size_bits) {
              Console.WriteLine("GPUVerify: error: _SIZE_T_TYPE size cannot be smaller than group_size_x size");
              Environment.Exit(1);
            }

            new ModSetCollector().DoModSetAnalysis(Program);

            CheckWellFormedness();

            GlobalArraySourceNames = new Dictionary<string,string>();
            foreach(var g in Program.TopLevelDeclarations.OfType<GlobalVariable>()) {
                string sourceName = QKeyValue.FindStringAttribute(g.Attributes, "source_name");
                if (sourceName != null)
                  GlobalArraySourceNames[g.Name] = sourceName;
            }

            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
            {
                this.NoAccessInstrumenter = new NoAccessInstrumenter(this);
            }

            if (GPUVerifyVCGenCommandLineOptions.ConstantWriteChecks) {
                this.ConstantWriteInstrumenter = new ConstantWriteInstrumenter(this);
            }
            if (GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
            {
              this.RaceInstrumenter = new NullRaceInstrumenter();
            } else {
              if (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.STANDARD) {
                this.RaceInstrumenter = new StandardRaceInstrumenter(this);
              } else {
                Debug.Assert(
                  RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_SINGLE ||
                  RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.WATCHDOG_MULTIPLE);
                this.RaceInstrumenter = new WatchdogRaceInstrumenter(this);
              }
            }
        }

        private void CheckWellFormedness()
        {
            int errorCount = Check();
            if (errorCount != 0)
            {
                Console.WriteLine("{0} GPUVerify format errors detected in {1}", errorCount, GPUVerifyVCGenCommandLineOptions.inputFiles[GPUVerifyVCGenCommandLineOptions.inputFiles.Count - 1]);
                Environment.Exit(1);
            }

            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
            {
              foreach (var b in Program.Blocks()) {
                foreach (var c in b.Cmds.OfType<CallCmd>()) {
                  if (QKeyValue.FindBoolAttribute(c.Attributes,"atomic")) {
                    Console.WriteLine("GPUVerify: error: --equality-abstraction cannot be used with atomics.");
                    Environment.Exit(1);
                  }
                }
              }
            }

            if (GPUVerifyVCGenCommandLineOptions.CheckSingleNonInlinedImpl)
            {
              var NonInlinedImpls = Program.Implementations().Where(
                Item => QKeyValue.FindIntAttribute((Item as Implementation).Attributes, "inline", -1) == -1);
              if (NonInlinedImpls.Count() != 1)
              {
                  Console.WriteLine("GPUVerify: warning: Found {0} non-inlined implementations.", NonInlinedImpls.Count());
                  foreach (Implementation impl in NonInlinedImpls)
                  {
                      Console.WriteLine("  {0}", impl.Name);
                  }
              }
            }
        }

        private int GetSizeTBits()
        {
            var candidates = Program.TopLevelDeclarations.OfType<TypeSynonymDecl>().
              Where(Item => Item.Name == _SIZE_T_BITS_TYPE);
            if (candidates.Count() != 1 || !candidates.ToList()[0].Body.IsBv) {
                Console.WriteLine("GPUVerify: error: exactly one _SIZE_T_TYPE bit-vector type must be specified");
                Environment.Exit(1);
            }
            return candidates.ToList()[0].Body.BvBits;
        }

        private int GetIdSizeBits()
        {
            var candidates = Program.TopLevelDeclarations.OfType<Constant>().
              Where(Item => Item.Name == GROUP_SIZE_X_STRING);
            if (candidates.Count() != 1) {
                Console.WriteLine("GPUVerify: error: exactly one group_size_x must be specified");
                Environment.Exit(1);
            }
            if (candidates.ToList()[0].TypedIdent.Type.IsInt)
                return this.size_t_bits; // Number of bits is irrelevant
            if (!candidates.ToList()[0].TypedIdent.Type.IsBv) {
                Console.WriteLine("GPUVerify: error: group_size_x must be of type int or bv");
                Environment.Exit(1);
            }
            return candidates.ToList()[0].TypedIdent.Type.BvBits;
        }

        private Dictionary<Procedure, Implementation> GetKernelProcedures()
        {
          var Result = new Dictionary<Procedure, Implementation>();
          foreach (Declaration D in Program.TopLevelDeclarations) {
            if (QKeyValue.FindBoolAttribute(D.Attributes, "kernel")) {
              if (D is Implementation) {
                Result[(D as Implementation).Proc] = D as Implementation;
              }
              if (D is Procedure) {
                if (!Result.ContainsKey(D as Procedure)) {
                  Result[D as Procedure] = null;
                }
              }
            }
          }
          return Result;
        }

        private Procedure FindOrCreateBarrierProcedure()
        {
            var p = CheckSingleInstanceOfAttributedProcedure("barrier");
            if (p == null)
            {
                p = new Procedure(Token.NoToken, "barrier", new List<TypeVariable>(),
                                  new List<Variable>(new Variable[] {
                                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__local_fence", IntRep.GetIntType(1)), true),
                                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__global_fence", IntRep.GetIntType(1)), true) }),
                                  new List<Variable>(),
                                  new List<Requires>(), new List<IdentifierExpr>(),
                                  new List<Ensures>(),
                                  new QKeyValue(Token.NoToken, "barrier", new List<object>(), null));
                Program.TopLevelDeclarations.Add(p);
                ResContext.AddProcedure(p);
            }
            return p;
        }

        private Procedure FindOrCreateBarrierInvariantProcedure() {
          var p = CheckSingleInstanceOfAttributedProcedure("barrier_invariant");
          if (p == null) {
            p = new Procedure(Token.NoToken, "barrier_invariant", new List<TypeVariable>(),
                              new List<Variable>(new Variable[] {
                                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__cond",
                                      Microsoft.Boogie.Type.Bool), true)
                              }),
                              new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(),
                              new List<Ensures>(),
                              new QKeyValue(Token.NoToken, "barrier_invariant", new List<object>(), null));
            Program.TopLevelDeclarations.Add(p);
            ResContext.AddProcedure(p);
          }
          return p;
        }

        private Procedure FindOrCreateBarrierInvariantInstantiationProcedure() {
          var p = CheckSingleInstanceOfAttributedProcedure("barrier_invariant_instantiation");
          if (p == null) {
            p = new Procedure(Token.NoToken, "barrier_invariant_instantiation", new List<TypeVariable>(),
                              new List<Variable>(new Variable[] {
                                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__t1",
                                      IntRep.GetIntType(size_t_bits)), true),
                                    new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "__t2",
                                      IntRep.GetIntType(size_t_bits)), true)
                              }),
                              new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(),
                              new List<Ensures>(),
                              new QKeyValue(Token.NoToken, "barrier_invariant_instantiation", new List<object>(), null));
            Program.TopLevelDeclarations.Add(p);
            ResContext.AddProcedure(p);
          }
          return p;
        }

        private Procedure CheckSingleInstanceOfAttributedProcedure(string attribute)
        {
            Procedure attributedProcedure = null;

            foreach (Declaration decl in Program.TopLevelDeclarations)
            {
                if (!QKeyValue.FindBoolAttribute(decl.Attributes, attribute))
                {
                    continue;
                }

                if (decl is Implementation)
                {
                    continue;
                }

                if (decl is Procedure)
                {
                    if (attributedProcedure == null)
                    {
                        attributedProcedure = decl as Procedure;
                    }
                    else
                    {
                        Error(decl, "\"{0}\" attribute specified for procedure {1}, but it has already been specified for procedure {2}", attribute, (decl as Procedure).Name, attributedProcedure.Name);
                    }

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
                first.line, first.col - 1);
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
                constFieldRef = new Constant(Token.NoToken,
                  new TypedIdent(Token.NoToken, attr, IntRep.GetIntType(id_size_bits)), /*unique=*/false);
                constFieldRef.AddAttribute(attr);
                Program.TopLevelDeclarations.Add(constFieldRef);
            }
        }

        private bool FindNonLocalVariables()
        {
            bool success = true;
            foreach (Declaration D in Program.TopLevelDeclarations)
            {
                if (D is Variable &&
                    (D as Variable).IsMutable &&
                    (D as Variable).TypedIdent.Type is MapType &&
                    !ReservedNames.Contains((D as Variable).Name))
                {
                    if (QKeyValue.FindBoolAttribute(D.Attributes, "group_shared"))
                    {
                        KernelArrayInfo.getGroupSharedArrays().Add(D as Variable);
                    }
                    else if (QKeyValue.FindBoolAttribute(D.Attributes, "global"))
                    {
                        KernelArrayInfo.getGlobalArrays().Add(D as Variable);
                    }
                    else if (QKeyValue.FindBoolAttribute(D.Attributes, "constant"))
                    {
                        KernelArrayInfo.getConstantArrays().Add(D as Variable);
                    }
                    else
                    {
                      if (!QKeyValue.FindBoolAttribute(D.Attributes, "atomic_usedmap")) {
                        KernelArrayInfo.getPrivateArrays().Add(D as Variable);
                      }
                    }
                }
                else if (D is Constant)
                {
                    Constant C = D as Constant;

                    success &= SetConstAttributeField(C, LOCAL_ID_X_STRING, ref _X);
                    success &= SetConstAttributeField(C, LOCAL_ID_Y_STRING, ref _Y);
                    success &= SetConstAttributeField(C, LOCAL_ID_Z_STRING, ref _Z);

                    success &= SetConstAttributeField(C, GROUP_SIZE_X_STRING, ref _GROUP_SIZE_X);
                    success &= SetConstAttributeField(C, GROUP_SIZE_Y_STRING, ref _GROUP_SIZE_Y);
                    success &= SetConstAttributeField(C, GROUP_SIZE_Z_STRING, ref _GROUP_SIZE_Z);

                    success &= SetConstAttributeField(C, GROUP_ID_X_STRING, ref _GROUP_X);
                    success &= SetConstAttributeField(C, GROUP_ID_Y_STRING, ref _GROUP_Y);
                    success &= SetConstAttributeField(C, GROUP_ID_Z_STRING, ref _GROUP_Z);

                    success &= SetConstAttributeField(C, NUM_GROUPS_X_STRING, ref _NUM_GROUPS_X);
                    success &= SetConstAttributeField(C, NUM_GROUPS_Y_STRING, ref _NUM_GROUPS_Y);
                    success &= SetConstAttributeField(C, NUM_GROUPS_Z_STRING, ref _NUM_GROUPS_Z);


                }
            }

            MaybeCreateAttributedConst(LOCAL_ID_X_STRING, ref _X);
            MaybeCreateAttributedConst(LOCAL_ID_Y_STRING, ref _Y);
            MaybeCreateAttributedConst(LOCAL_ID_Z_STRING, ref _Z);

            MaybeCreateAttributedConst(GROUP_SIZE_X_STRING, ref _GROUP_SIZE_X);
            MaybeCreateAttributedConst(GROUP_SIZE_Y_STRING, ref _GROUP_SIZE_Y);
            MaybeCreateAttributedConst(GROUP_SIZE_Z_STRING, ref _GROUP_SIZE_Z);

            MaybeCreateAttributedConst(GROUP_ID_X_STRING, ref _GROUP_X);
            MaybeCreateAttributedConst(GROUP_ID_Y_STRING, ref _GROUP_Y);
            MaybeCreateAttributedConst(GROUP_ID_Z_STRING, ref _GROUP_Z);

            MaybeCreateAttributedConst(NUM_GROUPS_X_STRING, ref _NUM_GROUPS_X);
            MaybeCreateAttributedConst(NUM_GROUPS_Y_STRING, ref _NUM_GROUPS_Y);
            MaybeCreateAttributedConst(NUM_GROUPS_Z_STRING, ref _NUM_GROUPS_Z);

            if(GPUVerifyVCGenCommandLineOptions.EliminateRedundantReadInstrumentation) {
              ComputeReadOnlyArrays();
            }

            return success;
        }

        private void ComputeReadOnlyArrays()
        {
          IEnumerable<Variable> WrittenArrays =
           Program.TopLevelDeclarations.OfType<Procedure>()
                  .Select(item => item.Modifies)
                  .SelectMany(Item => Item)
                  .Select(Item => Item.Decl)
                  .Where(item => KernelArrayInfo.ContainsNonLocalArray(item));
          foreach(var v in KernelArrayInfo.getAllNonLocalArrays().Where(
            Item => !WrittenArrays.Contains(Item))) {
            KernelArrayInfo.getReadOnlyNonLocalArrays().Add(v);
          }

        }

        private void CheckSpecialConstantType(Constant C)
        {
            if (!(C.TypedIdent.Type.IsInt || C.TypedIdent.Type.IsBv))
            {
                Error(C.tok, "Special constant '" + C.Name + "' must have type 'int' or 'bv'");
            }
        }

        private void MergeBlocksIntoPredecessors(bool UniformityMatters = true)
        {
            foreach (var impl in Program.Implementations())
                UniformityAnalyser.MergeBlocksIntoPredecessors(Program, impl,
                  UniformityMatters ? uniformityAnalyser : null);
        }

        internal void doit()
        {
            Microsoft.Boogie.CommandLineOptions.Clo.PrintUnstructured = 2;

            if(GPUVerifyVCGenCommandLineOptions.PrintLoopStatistics) {
              PrintLoopStatistics();
            }

            CheckUserSuppliedLoopInvariants();

            IdentifyArraysAccessedAsynchronously();

            DuplicateBarriers();

            if (GPUVerifyVCGenCommandLineOptions.IdentifySafeBarriers) {
              IdentifySafeBarriers();
            }

            if (!ProgramUsesBarrierInvariants()) {
                GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks = false;
            }

            if (GPUVerifyVCGenCommandLineOptions.RemovePrivateArrayAccesses) {
              EliminateLiteralIndexedPrivateArrays();
            }

            if (GPUVerifyVCGenCommandLineOptions.RefinedAtomics)
              RefineAtomicAbstraction();

            var nonUniformVars = new List<Variable> { _X, _Y, _Z };
            if(!GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking) {
                nonUniformVars.AddRange(new Variable[] { _GROUP_X, _GROUP_Y, _GROUP_Z } );
            }
            uniformityAnalyser = DoUniformityAnalysis(nonUniformVars);

            if (GPUVerifyVCGenCommandLineOptions.ShowUniformityAnalysis) {
                uniformityAnalyser.dump();
            }

            DoVariableDefinitionAnalysis();

            DoReducedStrengthAnalysis();

            DoMayBePowerOfTwoAnalysis();

            DoArrayControlFlowAnalysis();

            if (GPUVerifyVCGenCommandLineOptions.DoCallSiteAnalysis)
            {
                DoCallSiteAnalysis();
            }

            if (GPUVerifyVCGenCommandLineOptions.Inference)
            {

              foreach (var impl in Program.Implementations().ToList())
                {
                    if (!GPUVerifyVCGenCommandLineOptions.DisableInessentialLoopDetection)
                        LoopInvariantGenerator.EstablishDisabledLoops(this, impl);
                    LoopInvariantGenerator.PreInstrument(this, impl);
                }

                if (GPUVerifyVCGenCommandLineOptions.ShowStages) {
                  EmitProgram(outputFilename + "_pre_inference");
                }

            }

            if (GPUVerifyVCGenCommandLineOptions.ConstantWriteChecks) {
                ConstantWriteInstrumenter.AddConstantWriteInstrumentation();
            }

            if (GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams.Count > 0) {
              AddParamsAsPreconditions();
            }

            RaceInstrumenter.AddRaceCheckingInstrumentation();

            if (GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks)
            {
                NoAccessInstrumenter.AddNoAccessInstrumentation();
            }

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_instrumented");
            }

            AbstractSharedState();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_abstracted");
            }

            MergeBlocksIntoPredecessors();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_merged_pre_predication");
            }

            if (GPUVerifyVCGenCommandLineOptions.WarpSync && GPUVerifyVCGenCommandLineOptions.WarpMethod.Equals("resync"))
            {
              AddWarpSyncs();
            }

            MakeKernelPredicated();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_predicated");
            }

            MergeBlocksIntoPredecessors(false);

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_merged_post_predication");
            }

            MakeKernelDualised();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_dualised");
            }

            if (GPUVerifyVCGenCommandLineOptions.NonDeterminiseUninterpretedFunctions) {
              NonDeterminiseUninterpretedFunctions();
              if (GPUVerifyVCGenCommandLineOptions.ShowStages) {
                EmitProgram(outputFilename + "_ufs_removed");
              }
            }

            RaceInstrumenter.AddRaceCheckingDeclarations();

            foreach(var b in BarrierProcedures) {
              GenerateBarrierImplementation(b);
            }

            // We now do modset analysis here because the previous passes add new
            // global variables
            new ModSetCollector().DoModSetAnalysis(Program);

            if(GPUVerifyVCGenCommandLineOptions.OptimiseBarrierIntervals) {
              OptimiseBarrierIntervals();
            }

            GenerateStandardKernelContract();

            if (GPUVerifyVCGenCommandLineOptions.ShowStages)
            {
                EmitProgram(outputFilename + "_ready_to_verify");
            }

            if (GPUVerifyVCGenCommandLineOptions.Inference)
            {
              ComputeInvariant();

              if (GPUVerifyVCGenCommandLineOptions.AbstractHoudini)
              {
                new AbstractHoudiniTransformation(this).DoAbstractHoudiniTransform();
              }

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
              switch (GPUVerifyVCGenCommandLineOptions.WarpMethod) {
                case "resync" : GenerateWarpSyncs(); break;
                case "twopass" : if (GPUVerifyVCGenCommandLineOptions.NoWarp) DoNoWarp(); else DoOnlyWarp(); break;
              }
            }

            EmitProgram(outputFilename);

        }

        private void IdentifyArraysAccessedAsynchronously() {
          foreach(var AsyncCall in Program.Blocks().Select(Item => Item.Cmds).SelectMany(Item => Item).
            OfType<CallCmd>().Where(Item => QKeyValue.FindBoolAttribute(
              Item.Attributes, "async_work_group_copy"))) {
            Variable DstArray =
              (AsyncCall.Outs[1] as IdentifierExpr).Decl;
            Variable SrcArray =
              (AsyncCall.Ins[1] as IdentifierExpr).Decl;
            Debug.Assert(KernelArrayInfo.getAllNonLocalArrays().Contains(DstArray));
            Debug.Assert(KernelArrayInfo.getAllNonLocalArrays().Contains(SrcArray));
            ArraysAccessedByAsyncWorkGroupCopy[AccessType.WRITE].Add(DstArray.Name);
            ArraysAccessedByAsyncWorkGroupCopy[AccessType.READ].Add(SrcArray.Name);
          }
        }

        private int DeepestSubregion(IRegion r) {
          if(r.SubRegions().Where(Item => Item.Identifier() != r.Identifier()).Count() == 0) {
            return 0;
          }
          return 1 + r.SubRegions().Where(Item => Item.Identifier() != r.Identifier()).Select(Item => DeepestSubregion(Item)).Max();
        }

        private void PrintLoopStatistics() {
          // For each implementation, dump the number of loops and the depth of the deepest loop nest to a file
          var loopsOutputFile = Path.GetFileNameWithoutExtension(GPUVerifyVCGenCommandLineOptions.inputFiles[0]) + ".loops";
          using (TokenTextWriter writer = new TokenTextWriter(loopsOutputFile)) {
            foreach(var impl in Program.Implementations()) {
              writer.WriteLine("Implementation: " + impl.Name);
              writer.WriteLine("Number of loops: " + RootRegion(impl).SubRegions().Count());
              writer.WriteLine("Depth of deepest loop nest: " + DeepestSubregion(RootRegion(impl)));
              writer.WriteLine("");
            }
          }
        }

        private void AddParamsAsPreconditions()
        {
          List<string> param_values =
            GPUVerifyVCGenCommandLineOptions.KernelInterceptorParams;
          // Todo: work out how to locate the "original name",
          // rather than prepending the "$" like this.
          string target_name = "$" + param_values[0];

          // Locate the kernel with the given name
          bool found_flag = false;
          Procedure proc = null;
          foreach(KeyValuePair<Procedure, Implementation> entry
                  in KernelProcedures) {
            if (target_name == entry.Key.Name) {
              // Console.WriteLine("Found kernel " + target_name + ".");
              found_flag = true;
              proc = entry.Key;
              break;
            }
          }
          if (found_flag == false) {
            Console.WriteLine("Error: Couldn't find kernel "
                              + target_name + ".");
            Environment.Exit(1);
          }

          // Fail if too many params given (note that first
          // element of param_values is the name of the kernel)
          if (param_values.Count - 1 > proc.InParams.Count) {
            Console.WriteLine("Error: Too many parameter values.");
            Environment.Exit(1);
          }

          // Create requires clauses
          for (int ctr = 1; ctr < param_values.Count; ctr++) {
            Variable v = proc.InParams[ctr-1];
            Expr v_expr = new IdentifierExpr(v.tok, v);
            string val = param_values[ctr];
            // Asterisk used to signify arbitrary value,
            // hence no requires clause needed.
            if (val=="*") continue;

	    BigInteger arg = BigInteger.Parse(val,NumberStyles.HexNumber);

            Expr val_expr =
		    IntRep.GetLiteral(arg, ((BvType)v.TypedIdent.Type).Bits);
            Expr v_eq_val = Expr.Eq(v_expr, val_expr);
            proc.Requires.Add(new Requires(false, v_eq_val));
            // Console.WriteLine("__requires(" + v.Name + "==" + val + ")");
          }
        }

        private void IdentifySafeBarriers()
        {
          var uni = DoUniformityAnalysis(new List<Variable> { _X, _Y, _Z });
          foreach(var b in BarrierProcedures) {
            if(uni.IsUniform(b.Name)) {
              b.AddAttribute("safe_barrier", new object[] { });
            }
          }
        }

        private void DuplicateBarriers()
        {
          // Make a separate barrier procedure for every barrier call.
          // This paves the way for barrier divergence optimisations
          // for specific barriers
          Contract.Requires(BarrierProcedures.Count() == 1);
          Program.TopLevelDeclarations.Remove(BarrierProcedures.ToList()[0]);
          BarrierProcedures = new HashSet<Procedure>();
          int BarrierCounter = 0;
          foreach(Block b in Program.Blocks().ToList()) {
            List<Cmd> newCmds = new List<Cmd>();
            foreach(Cmd c in b.Cmds) {
              var call = c as CallCmd;
              if(call == null || !IsBarrier(call.Proc)) {
                newCmds.Add(c);
                continue;
              }
              Procedure NewBarrier = new Duplicator().VisitProcedure(call.Proc);
              Debug.Assert(IsBarrier(NewBarrier));
              NewBarrier.Name = NewBarrier.Name + "_duplicated_" + BarrierCounter;
              BarrierCounter++;
              var NewCall = new CallCmd(call.tok, NewBarrier.Name, call.Ins, call.Outs, call.Attributes);
              NewCall.Proc = NewBarrier;
              newCmds.Add(NewCall);
              Program.TopLevelDeclarations.Add(NewBarrier);
              BarrierProcedures.Add(NewBarrier);
              ResContext.AddProcedure(NewBarrier);
            }
            b.Cmds = newCmds;
          }
        }

        private void NonDeterminiseUninterpretedFunctions()
        {
          var UFRemover = new UninterpretedFunctionRemover();
          UFRemover.Eliminate(Program);
        }

        private void EliminateLiteralIndexedPrivateArrays()
        {
          // If a program contains private arrays that are only ever indexed by
          // literals, these can be eliminated.  This reduces the extent to which
          // arrays are used in the generated .bpl program, which may benefit
          // constraint solving.
          var Eliminator = new LiteralIndexedArrayEliminator(this);
          Eliminator.Eliminate(Program);
        }

        private void AddCaptureStates()
        {
          AddCaptureStatesToLoops();
          AddCaptureStatesAfterProcedureCalls();
        }

        private void AddCaptureStatesAfterProcedureCalls()
        {
          int counter = 0;
          foreach(var b in Program.Blocks()) {
            List<Cmd> NewCmds = new List<Cmd>();
            foreach(var c in b.Cmds) {
              NewCmds.Add(c);
              var call = c as CallCmd;
              if(call != null && !ProcedureIsInlined(call.Proc)) {
                NewCmds.Add(new AssumeCmd(Token.NoToken, Expr.True,
                  new QKeyValue(Token.NoToken, "captureState", new List<object> { "call_return_state_" + counter },
                    new QKeyValue(Token.NoToken, "procedureName", new List<object> { call.callee }, null))));
              }
            }
            b.Cmds = NewCmds;
          }
        }

        private void AddCaptureStatesToLoops()
        {
          // Add the ability to get the state right before entering each loop,
          // at the loop head itself, and right before taking a back-edge
          int LoopCounter = 0;
          foreach(var impl in Program.Implementations()) {
            var CFG = Program.GraphFromImpl(impl);
            CFG.ComputeLoops();
            foreach(var Header in CFG.Headers) {
              AddStateCaptureToLoopHead(LoopCounter, Header);
              AppendStateCaptureToBlocks("loop_back_edge_state", LoopCounter, CFG.BackEdgeNodes(Header));
              AppendStateCaptureToBlocks("loop_entry_state", LoopCounter, LoopEntryEdgeNodes(CFG, Header));
              LoopCounter++;
            }
          }
        }

        private static IEnumerable<Block> LoopEntryEdgeNodes(Microsoft.Boogie.GraphUtil.Graph<Block> CFG, Block Header) {
          return CFG.Predecessors(Header).Where(Item => !CFG.BackEdgeNodes(Header).Contains(Item));
        }

        private void AppendStateCaptureToBlocks(string StateNamePrefix, int LoopCounter, IEnumerable<Block> Blocks) {
          int Counter = 0;
          foreach (var n in Blocks) {
            n.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.True,
              new QKeyValue(Token.NoToken, "captureState", new List<object> { StateNamePrefix + "_" + LoopCounter + "_" + Counter }, null)));
            Counter++;
          }
        }

        private static void AddStateCaptureToLoopHead(int LoopCounter, Block b) {
          List<Cmd> NewCmds = new List<Cmd>();
          NewCmds.Add(new AssumeCmd(Token.NoToken, Expr.True,
            new QKeyValue(Token.NoToken, "captureState", new List<object> { "loop_head_state_" + LoopCounter }, null)));
          NewCmds.AddRange(b.Cmds);
          b.Cmds = NewCmds;
        }

        private void CheckUserSuppliedLoopInvariants()
        {
          foreach(var impl in Program.Implementations()) {
            var blockGraph = Program.ProcessLoops(impl);
            foreach(var b in impl.Blocks) {
              bool ValidPositionForInvariant = blockGraph.Headers.Contains(b);
              foreach(var c in b.Cmds) {
                var pc = c as PredicateCmd;
                if(pc != null) {
                  if(QKeyValue.FindBoolAttribute(pc.Attributes, "originated_from_invariant")
                    && !ValidPositionForInvariant) {
                    var SourceLoc = new SourceLocationInfo(pc.Attributes, GPUVerifyVCGenCommandLineOptions.inputFiles[0], pc.tok);

                    Console.Write("\n" + SourceLoc.Top() + ": ");
                    Console.WriteLine("user-specified invariant does not appear at loop head.");
                    Console.WriteLine("\nNote: a common cause of this is due to the use of short-circuit operations;");
                    Console.WriteLine("      these should not be used in invariants.");
                    Environment.Exit(1);
                  }
                } else {
                  ValidPositionForInvariant = false;
                }
              }
            }
          }
        }

        private void OptimiseBarrierIntervals()
        {
          var BarrierIntervalsAnalysis = new BarrierIntervalsAnalysis(this, BarrierStrength.GROUP_SHARED);
          BarrierIntervalsAnalysis.Compute();
          BarrierIntervalsAnalysis.RemoveRedundantReads();
        }

        private void DoMayBePowerOfTwoAnalysis()
        {
            mayBePowerOfTwoAnalyser = new MayBePowerOfTwoAnalyser(this);
            mayBePowerOfTwoAnalyser.Analyse();
            relationalPowerOfTwoAnalyser = new RelationalPowerOfTwoAnalyser(this);
            relationalPowerOfTwoAnalyser.Analyse();
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
            if (GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis) {
              foreach (Implementation i in KernelProcedures.Values) {
                if (i != null) {
                  entryPoints.Add(i);
                }
              }
            }

            var result = new UniformityAnalyser(Program, GPUVerifyVCGenCommandLineOptions.DoUniformityAnalysis,
                                                        entryPoints, nonUniformVars);
            result.Analyse();
            return result;
        }

        private void DoVariableDefinitionAnalysis()
        {
            varDefAnalyses = Program.Implementations()
                .ToDictionary(i => i, i => VariableDefinitionAnalysis.Analyse(this, i));
        }

        private void DoReducedStrengthAnalysis()
        {
            reducedStrengthAnalyses = Program.Implementations()
                .ToDictionary(i => i, i => ReducedStrengthAnalysis.Analyse(this, i));
        }

        private void EmitProgram(string filename)
        {
          GVUtil.IO.EmitProgram(Program, filename);
        }

        private void ComputeInvariant()
        {
            foreach (var Impl in Program.Implementations().ToList())
            {
                LoopInvariantGenerator.PostInstrument(this, Impl);
                if (ProcedureIsInlined(Impl.Proc) || KernelProcedures.ContainsKey(Impl.Proc))
                {
                    continue;
                }
                AddCandidateRequires(Impl.Proc);
                RaceInstrumenter.AddRaceCheckingCandidateRequires(Impl.Proc);
                AddCandidateEnsures(Impl.Proc);
                RaceInstrumenter.AddRaceCheckingCandidateEnsures(Impl.Proc);
            }
        }

        private void AddCandidateEnsures(Procedure Proc)
        {
            HashSet<string> names = new HashSet<String>();
            foreach (Variable v in Proc.OutParams)
            {
                names.Add(GVUtil.StripThreadIdentifier(v.Name));
            }

            foreach (string name in names)
            {
                if (!uniformityAnalyser.IsUniform(Proc.Name, name))
                {
                    AddEqualityCandidateEnsures(Proc, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, name, Microsoft.Boogie.Type.Int)));
                }
            }

        }

        private void AddCandidateRequires(Procedure Proc)
        {
            HashSet<string> names = new HashSet<String>();
            foreach (Variable v in Proc.InParams)
            {
                names.Add(GVUtil.StripThreadIdentifier(v.Name));
            }

            foreach (string name in names)
            {

                if (IsPredicateOrTemp(name))
                {
                    Debug.Assert(name.Equals("_P"));
                    Debug.Assert(!uniformityAnalyser.IsUniform(Proc.Name));
                    AddCandidateRequires(Proc, Expr.Eq(
                        new IdentifierExpr(Proc.tok, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, name + "$1", Microsoft.Boogie.Type.Bool))),
                        new IdentifierExpr(Proc.tok, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, name + "$2", Microsoft.Boogie.Type.Bool)))
                    ), InferenceStages.BASIC_CANDIDATE_STAGE);
                }
                else
                {
                    if (!uniformityAnalyser.IsUniform(Proc.Name, name))
                    {
                        if (!uniformityAnalyser.IsUniform(Proc.Name))
                        {
                            AddPredicatedEqualityCandidateRequires(Proc, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, name, Microsoft.Boogie.Type.Int)));
                        }
                        AddEqualityCandidateRequires(Proc, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, name, Microsoft.Boogie.Type.Int)));
                    }
                }
            }

        }

        private void AddPredicatedEqualityCandidateRequires(Procedure Proc, Variable v)
        {
            AddCandidateRequires(Proc, Expr.Imp(
                Expr.And(
                    new IdentifierExpr(Proc.tok, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, "_P$1", Microsoft.Boogie.Type.Bool))),
                    new IdentifierExpr(Proc.tok, new LocalVariable(Proc.tok, new TypedIdent(Proc.tok, "_P$2", Microsoft.Boogie.Type.Bool)))
                ),
                Expr.Eq(
                    new IdentifierExpr(Proc.tok, new VariableDualiser(1, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable)),
                    new IdentifierExpr(Proc.tok, new VariableDualiser(2, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable))
                )
            ), InferenceStages.BASIC_CANDIDATE_STAGE);
        }

        private void AddEqualityCandidateRequires(Procedure Proc, Variable v)
        {
            AddCandidateRequires(Proc,
                Expr.Eq(
                    new IdentifierExpr(Proc.tok, new VariableDualiser(1, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable)),
                    new IdentifierExpr(Proc.tok, new VariableDualiser(2, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable))
                ), InferenceStages.BASIC_CANDIDATE_STAGE
            );
        }

        private void AddEqualityCandidateEnsures(Procedure Proc, Variable v)
        {
            AddCandidateEnsures(Proc,
                Expr.Eq(
                    new IdentifierExpr(Proc.tok, new VariableDualiser(1, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable)),
                    new IdentifierExpr(Proc.tok, new VariableDualiser(2, uniformityAnalyser, Proc.Name).VisitVariable(v.Clone() as Variable))
                ), InferenceStages.BASIC_CANDIDATE_STAGE);
        }

        internal void AddCandidateRequires(Procedure Proc, Expr e, int StageId)
        {
            Constant ExistentialBooleanConstant = Program.MakeExistentialBoolean(StageId);
            IdentifierExpr ExistentialBoolean = new IdentifierExpr(Proc.tok, ExistentialBooleanConstant);
            Proc.Requires.Add(new Requires(false, Expr.Imp(ExistentialBoolean, e)));
        }

        internal void AddCandidateEnsures(Procedure Proc, Expr e, int StageId)
        {
            Constant ExistentialBooleanConstant = Program.MakeExistentialBoolean(StageId);
            IdentifierExpr ExistentialBoolean = new IdentifierExpr(Proc.tok, ExistentialBooleanConstant);
            Proc.Ensures.Add(new Ensures(false, Expr.Imp(ExistentialBoolean, e)));
        }

        internal bool ContainsNamedVariable(HashSet<Variable> variables, string name)
        {
            foreach(Variable v in variables)
            {
                if(GVUtil.StripThreadIdentifier(v.Name) == name)
                {
                    return true;
                }
            }
            return false;
        }

        internal static bool IsPredicate(string v) {
          if (v.Length < 2) {
            return false;
          }
          if (!v.Substring(0, 1).Equals("p")) {
            return false;
          }
          for (int i = 1; i < v.Length; i++) {
            if (!Char.IsDigit(v.ToCharArray()[i])) {
              return false;
            }
          }
          return true;
        }

        internal static bool IsPredicateOrTemp(string lv) {

          // We should improve this by having a general convention
          // for names of temporary or predicate variables

          if (lv.Length >= 2) {
            if (lv.Substring(0, 1).Equals("p") || lv.Substring(0, 1).Equals("v")) {
              for (int i = 1; i < lv.Length; i++) {
                if (!Char.IsDigit(lv.ToCharArray()[i])) {
                  return false;
                }
              }
              return true;
            }

          }

          if (lv.Contains("_HAVOC_")) {
            return true;
          }

          return (lv.Length >= 2 && lv.Substring(0,2).Equals("_P")) ||
                  (lv.Length > 3 && lv.Substring(0,3).Equals("_LC")) ||
                  (lv.Length > 5 && lv.Substring(0,5).Equals("_temp"));
        }

        public static Microsoft.Boogie.Type GetTypeOfIdX()
        {
            Contract.Requires(_X != null);
            return _X.TypedIdent.Type;
        }

        public static Microsoft.Boogie.Type GetTypeOfIdY()
        {
            Contract.Requires(_Y != null);
            return _Y.TypedIdent.Type;
        }

        public static Microsoft.Boogie.Type GetTypeOfIdZ()
        {
            Contract.Requires(_Z != null);
            return _Z.TypedIdent.Type;
        }

        public static Microsoft.Boogie.Type GetTypeOfId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X")) return GetTypeOfIdX();
            if (dimension.Equals("Y")) return GetTypeOfIdY();
            if (dimension.Equals("Z")) return GetTypeOfIdZ();
            Debug.Assert(false);
            return null;
        }

        public bool KernelHasIdX()
        {
            return _X != null;
        }

        public bool KernelHasIdY()
        {
            return _Y != null;
        }

        public bool KernelHasIdZ()
        {
            return _Z != null;
        }

        public bool KernelHasGroupIdX()
        {
            return _GROUP_X != null;
        }

        public bool KernelHasGroupIdY()
        {
            return _GROUP_Y != null;
        }

        public bool KernelHasGroupIdZ()
        {
            return _GROUP_Z != null;
        }

        public bool KernelHasNumGroupsX()
        {
            return _NUM_GROUPS_X != null;
        }

        public bool KernelHasNumGroupsY()
        {
            return _NUM_GROUPS_Y != null;
        }

        public bool KernelHasNumGroupsZ()
        {
            return _NUM_GROUPS_Z != null;
        }

        public bool KernelHasGroupSizeX()
        {
            return _GROUP_SIZE_X != null;
        }

        public bool KernelHasGroupSizeY()
        {
            return _GROUP_SIZE_Y != null;
        }

        public bool KernelHasGroupSizeZ()
        {
            return _GROUP_SIZE_Z != null;
        }

        private void GenerateStandardKernelContract()
        {
            RaceInstrumenter.AddKernelPrecondition();

            GeneratePreconditionsForDimension("X");
            GeneratePreconditionsForDimension("Y");
            GeneratePreconditionsForDimension("Z");

            foreach (Declaration D in Program.TopLevelDeclarations)
            {
                if (!(D is Procedure))
                {
                    continue;
                }
                Procedure Proc = D as Procedure;
                if (ProcedureIsInlined(Proc) || ProcedureHasNoImplementation(Proc))
                {
                    continue;
                }

                Expr DistinctLocalIds =
                    Expr.Or(
                        Expr.Or(
                            Expr.Neq(
                            new IdentifierExpr(Token.NoToken, MakeThreadId("X", 1)),
                            new IdentifierExpr(Token.NoToken, MakeThreadId("X", 2))
                            ),
                            Expr.Neq(
                            new IdentifierExpr(Token.NoToken, MakeThreadId("Y", 1)),
                            new IdentifierExpr(Token.NoToken, MakeThreadId("Y", 2))
                            )
                        ),
                        Expr.Neq(
                        new IdentifierExpr(Token.NoToken, MakeThreadId("Z", 1)),
                        new IdentifierExpr(Token.NoToken, MakeThreadId("Z", 2))
                        )
                    );

                Proc.Requires.Add(new Requires(false, Expr.Imp(ThreadsInSameGroup(), DistinctLocalIds)));

                if (GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking)
                {
                    Proc.Requires.Add(new Requires(false, ThreadsInSameGroup()));
                }

                if (KernelProcedures.ContainsKey(Proc))
                {
                    bool foundNonUniform = false;
                    int indexOfFirstNonUniformParameter;
                    for (indexOfFirstNonUniformParameter = 0; indexOfFirstNonUniformParameter < Proc.InParams.Count(); indexOfFirstNonUniformParameter++)
                    {
                        if (!uniformityAnalyser.IsUniform(Proc.Name, GVUtil.StripThreadIdentifier(Proc.InParams[indexOfFirstNonUniformParameter].Name)))
                        {
                            foundNonUniform = true;
                            break;
                        }
                    }

                    if (foundNonUniform)
                    {
                        // I have a feeling this will never be reachable!!!
                        int numberOfNonUniformParameters = (Proc.InParams.Count() - indexOfFirstNonUniformParameter) / 2;
                        for (int i = indexOfFirstNonUniformParameter; i < numberOfNonUniformParameters; i++)
                        {
                            Proc.Requires.Add(new Requires(false,
                                Expr.Eq(new IdentifierExpr(Proc.InParams[i].tok, Proc.InParams[i]),
                                        new IdentifierExpr(Proc.InParams[i + numberOfNonUniformParameters].tok, Proc.InParams[i + numberOfNonUniformParameters]))));
                        }
                    }
                }

            }

        }

        internal bool ProcedureHasNoImplementation(Procedure Proc) {
          return !Program.Implementations().Select(i => i.Name).Contains(Proc.Name);
        }

        internal bool ProcedureIsInlined(Procedure Proc) {
          return QKeyValue.FindIntAttribute(Proc.Attributes, "inline", -1) == 1;
        }

        internal Expr ThreadsInSameWarp()
        {
            Expr warpsize = Expr.Ident(GPUVerifyVCGenCommandLineOptions.WarpSize + "bv" + id_size_bits, new BvType(id_size_bits));
            IEnumerable<Expr> tids = (new int[] {1,2}).Select(x => FlattenedThreadId(x));
            Expr[] sides = tids.Select(x => IntRep.MakeDiv(x,warpsize)).ToArray();
            return Expr.Eq(sides[0], sides[1]);
        }

        internal static Expr ThreadsInSameGroup()
        {
            if(GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking) {
              return Expr.True;
            }

            return Expr.And(
                                        Expr.And(
                                            Expr.Eq(
                                            new IdentifierExpr(Token.NoToken, MakeGroupId("X", 1)),
                                            new IdentifierExpr(Token.NoToken, MakeGroupId("X", 2))
                                            ),
                                            Expr.Eq(
                                            new IdentifierExpr(Token.NoToken, MakeGroupId("Y", 1)),
                                            new IdentifierExpr(Token.NoToken, MakeGroupId("Y", 2))
                                            )
                                        ),
                                        Expr.Eq(
                                        new IdentifierExpr(Token.NoToken, MakeGroupId("Z", 1)),
                                        new IdentifierExpr(Token.NoToken, MakeGroupId("Z", 2))
                                        )
                                    );
        }

        internal static int GetThreadSuffix(string p)
        {
            return Int32.Parse(p.Substring(p.IndexOf("$") + 1, p.Length - (p.IndexOf("$") + 1)));
        }

        internal LiteralExpr Zero(int bits) {
          return IntRep.GetLiteral(0, bits);
        }

        private void GeneratePreconditionsForDimension(String dimension)
        {
            foreach (Declaration D in Program.TopLevelDeclarations.ToList())
            {
                if (!(D is Procedure))
                {
                    continue;
                }
                Procedure Proc = D as Procedure;
                if (ProcedureIsInlined(Proc) || ProcedureHasNoImplementation(Proc))
                {
                    continue;
                }

                Expr GroupSizePositive = IntRep.MakeSgt(new IdentifierExpr(Token.NoToken, GetGroupSize(dimension)), Zero(id_size_bits));
                Expr NumGroupsPositive = IntRep.MakeSgt(new IdentifierExpr(Token.NoToken, GetNumGroups(dimension)), Zero(id_size_bits));
                Expr GroupIdNonNegative = IntRep.MakeSge(new IdentifierExpr(Token.NoToken, GetGroupId(dimension)), Zero(id_size_bits));
                Expr GroupIdLessThanNumGroups = IntRep.MakeSlt(new IdentifierExpr(Token.NoToken, GetGroupId(dimension)), new IdentifierExpr(Token.NoToken, GetNumGroups(dimension)));

                Proc.Requires.Add(new Requires(false, GroupSizePositive));
                Proc.Requires.Add(new Requires(false, NumGroupsPositive));

                if(GPUVerifyVCGenCommandLineOptions.OnlyIntraGroupRaceChecking) {
                  Proc.Requires.Add(new Requires(false, GroupIdNonNegative));
                  Proc.Requires.Add(new Requires(false, GroupIdLessThanNumGroups));
                } else {
                  Proc.Requires.Add(new Requires(false, new VariableDualiser(1, null, null).VisitExpr(GroupIdNonNegative)));
                  Proc.Requires.Add(new Requires(false, new VariableDualiser(2, null, null).VisitExpr(GroupIdNonNegative)));
                  Proc.Requires.Add(new Requires(false, new VariableDualiser(1, null, null).VisitExpr(GroupIdLessThanNumGroups)));
                  Proc.Requires.Add(new Requires(false, new VariableDualiser(2, null, null).VisitExpr(GroupIdLessThanNumGroups)));
                }

                Expr ThreadIdNonNegative = IntRep.MakeSge(new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)), Zero(id_size_bits));
                Expr ThreadIdLessThanGroupSize = IntRep.MakeSlt(new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)),
                  new IdentifierExpr(Token.NoToken, GetGroupSize(dimension)));

                Proc.Requires.Add(new Requires(false, new VariableDualiser(1, null, null).VisitExpr(ThreadIdNonNegative)));
                Proc.Requires.Add(new Requires(false, new VariableDualiser(2, null, null).VisitExpr(ThreadIdNonNegative)));
                Proc.Requires.Add(new Requires(false, new VariableDualiser(1, null, null).VisitExpr(ThreadIdLessThanGroupSize)));
                Proc.Requires.Add(new Requires(false, new VariableDualiser(2, null, null).VisitExpr(ThreadIdLessThanGroupSize)));

            }

        }

        internal Function GetOrCreateBVFunction(string functionName, string smtName, Microsoft.Boogie.Type resultType, params Microsoft.Boogie.Type[] argTypes)
        {
            Function f = (Function)ResContext.LookUpProcedure(functionName);
            if (f != null)
                return f;

            f = new Function(Token.NoToken, functionName,
                              new List<Variable>(argTypes.Select(t => new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", t))).ToArray()),
                              new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", resultType)));
            f.AddAttribute("bvbuiltin", smtName);
            Program.TopLevelDeclarations.Add(f);
            ResContext.AddProcedure(f);
            return f;
        }

        internal Function GetOrCreateIntFunction(string functionName, BinaryOperator.Opcode infixOp, Microsoft.Boogie.Type resultType, Microsoft.Boogie.Type lhsType, Microsoft.Boogie.Type rhsType)
        {
          Function f = (Function)ResContext.LookUpProcedure(functionName);
          if (f != null)
              return f;

          List<Variable> inParams = new List<Variable>();
          Variable lhs = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "x", lhsType));
          Variable rhs = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "y", rhsType));
          inParams.Add(lhs);
          inParams.Add(rhs);

          f = new Function(Token.NoToken, functionName,
                            inParams,
                            new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", resultType)));
          f.AddAttribute("inline", Expr.True);
          f.Body = Expr.Binary(infixOp, new IdentifierExpr(Token.NoToken, lhs), new IdentifierExpr(Token.NoToken, rhs)); 

          Program.TopLevelDeclarations.Add(f);
          ResContext.AddProcedure(f);
          return f;
        }

        internal Function GetOrCreateBinaryUF(string functionName, Microsoft.Boogie.Type resultType, Microsoft.Boogie.Type lhsType, Microsoft.Boogie.Type rhsType)
        {
          Function f = (Function)ResContext.LookUpProcedure(functionName);
          if (f != null)
              return f;

          f = new Function(Token.NoToken, functionName,
                            new List<Variable> { new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", lhsType)),
                                              new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", rhsType))},
                            new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", resultType)));
          Program.TopLevelDeclarations.Add(f);
          ResContext.AddProcedure(f);
          return f;
        }

        internal Constant GetGroupSize(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X")) return _GROUP_SIZE_X;
            if (dimension.Equals("Y")) return _GROUP_SIZE_Y;
            if (dimension.Equals("Z")) return _GROUP_SIZE_Z;
            Debug.Assert(false);
            return null;
        }

        internal Constant GetNumGroups(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X")) return _NUM_GROUPS_X;
            if (dimension.Equals("Y")) return _NUM_GROUPS_Y;
            if (dimension.Equals("Z")) return _NUM_GROUPS_Z;
            Debug.Assert(false);
            return null;
        }

        internal static Constant MakeThreadId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            string name = null;
            if (dimension.Equals("X")) name = _X.Name;
            if (dimension.Equals("Y")) name = _Y.Name;
            if (dimension.Equals("Z")) name = _Z.Name;
            Debug.Assert(name != null);
            return new Constant(Token.NoToken, new TypedIdent(Token.NoToken, name, GetTypeOfId(dimension)));
        }

        internal static Constant MakeThreadId(string dimension, int number)
        {
            Constant resultWithoutThreadId = MakeThreadId(dimension);
            return new Constant(Token.NoToken, new TypedIdent(
              Token.NoToken, resultWithoutThreadId.Name + "$" + number, GetTypeOfId(dimension)));
        }

        internal static Constant GetGroupId(string dimension)
        {
            Contract.Requires(dimension.Equals("X") || dimension.Equals("Y") || dimension.Equals("Z"));
            if (dimension.Equals("X")) return _GROUP_X;
            if (dimension.Equals("Y")) return _GROUP_Y;
            if (dimension.Equals("Z")) return _GROUP_Z;
            Debug.Assert(false);
            return null;
        }

        internal static Constant MakeGroupId(string dimension, int number)
        {
            Constant resultWithoutThreadId = GetGroupId(dimension);
            return new Constant(Token.NoToken, new TypedIdent(Token.NoToken, resultWithoutThreadId.Name + "$" + number, GetTypeOfId(dimension)));
        }

        private void GenerateBarrierImplementation(Procedure BarrierProcedure)
        {
            List<BigBlock> bigblocks = new List<BigBlock>();
            BigBlock barrierEntryBlock = new BigBlock(Token.NoToken, "__BarrierImpl", new List<Cmd>(), null, null);
            bigblocks.Add(barrierEntryBlock);

            Expr P1 = null, P2 = null, LocalFence1 = null, LocalFence2 = null, GlobalFence1 = null, GlobalFence2 = null;

            if (uniformityAnalyser.IsUniform(BarrierProcedure.Name)) {
              P1 = Expr.True;
              P2 = Expr.True;
            }

            foreach(Formal f in BarrierProcedure.InParams) {
              int Thread;
              string name = GVUtil.StripThreadIdentifier(f.Name, out Thread);
              if(name.Equals(BarrierProcedureLocalFenceArgName)) {
                if (uniformityAnalyser.IsUniform(BarrierProcedure.Name, name)) {
                  LocalFence1 = MakeFenceExpr(f);
                  LocalFence2 = MakeFenceExpr(f);
                }
                else {
                  if (Thread == 1) {
                    LocalFence1 = MakeFenceExpr(f);
                  }
                  else {
                    Debug.Assert(Thread == 2);
                    LocalFence2 = MakeFenceExpr(f);
                  }
                }
              }
              else if (name.Equals(BarrierProcedureGlobalFenceArgName)) {
                if (uniformityAnalyser.IsUniform(BarrierProcedure.Name, name)) {
                  GlobalFence1 = MakeFenceExpr(f);
                  GlobalFence2 = MakeFenceExpr(f);
                }
                else {
                  if (Thread == 1) {
                    GlobalFence1 = MakeFenceExpr(f);
                  }
                  else {
                    Debug.Assert(Thread == 2);
                    GlobalFence2 = MakeFenceExpr(f);
                  }
                }
              }
              else {
                Debug.Assert(name.Equals("_P"));
                if (Thread == 1) {
                  P1 = new IdentifierExpr(Token.NoToken, f);
                }
                else {
                  Debug.Assert(Thread == 2);
                  P2 = new IdentifierExpr(Token.NoToken, f);
                }
              }
            }

            Debug.Assert(P1 != null);
            Debug.Assert(P2 != null);
            Debug.Assert(LocalFence1 != null);
            Debug.Assert(LocalFence2 != null);
            Debug.Assert(GlobalFence1 != null);
            Debug.Assert(GlobalFence2 != null);

            if(!QKeyValue.FindBoolAttribute(BarrierProcedure.Attributes, "safe_barrier")) {
              Expr DivergenceCondition = Expr.Imp(ThreadsInSameGroup(), Expr.Eq(P1, P2));
              Requires nonDivergenceRequires = new Requires(false, DivergenceCondition);
              nonDivergenceRequires.Attributes = new QKeyValue(Token.NoToken, "barrier_divergence",
                new List<object>(new object[] { }), null);
              BarrierProcedure.Requires.Add(nonDivergenceRequires);
            }

            if (!GPUVerifyVCGenCommandLineOptions.OnlyDivergence)
            {
                List<BigBlock> returnbigblocks = new List<BigBlock>();
                returnbigblocks.Add(new BigBlock(Token.NoToken, "__Disabled", new List<Cmd>(), null, new ReturnCmd(Token.NoToken)));
                StmtList returnstatement = new StmtList(returnbigblocks, BarrierProcedure.tok);

                Expr IfGuard = Expr.Or(Expr.And(Expr.Not(P1), Expr.Not(P2)), Expr.And(ThreadsInSameGroup(), Expr.Or(Expr.Not(P1), Expr.Not(P2))));
                barrierEntryBlock.ec = new IfCmd(Token.NoToken, IfGuard, returnstatement, null, null);
            }

            var SharedArrays = KernelArrayInfo.getGroupSharedArrays();
            SharedArrays = SharedArrays.Where(x => !KernelArrayInfo.getReadOnlyNonLocalArrays().Contains(x)).ToList();
            if(SharedArrays.Count > 0) {

                bigblocks.AddRange(
                      MakeResetBlocks(Expr.And(P1, LocalFence1), SharedArrays));

                // This could be relaxed to take into account whether the threads are in different
                // groups, but for now we keep it relatively simple

                Expr AtLeastOneEnabledWithLocalFence =
                  Expr.Or(Expr.And(P1, LocalFence1), Expr.And(P2, LocalFence2));

                if (SomeArrayModelledNonAdversarially(SharedArrays)) {
                  var NoAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ? 
                    SharedArrays.Select(x => FindOrCreateNotAccessedVariable(x.Name, (x.TypedIdent.Type as MapType).Arguments[0])) :
                    Enumerable.Empty<Variable>();
                  var HavocVars = SharedArrays.Concat(NoAccessVars).ToList();
                  bigblocks.Add(new BigBlock(Token.NoToken, null, new List<Cmd>(),
                    new IfCmd(Token.NoToken,
                      AtLeastOneEnabledWithLocalFence,
                      new StmtList(MakeHavocBlocks(HavocVars), Token.NoToken), null, null), null));
                }
            }

            var GlobalArrays = KernelArrayInfo.getGlobalArrays();
            GlobalArrays = GlobalArrays.Where(x => !KernelArrayInfo.getReadOnlyNonLocalArrays().Contains(x)).ToList();
            if (GlobalArrays.Count > 0)
            {
                bigblocks.AddRange(
                      MakeResetBlocks(Expr.And(P1, GlobalFence1), GlobalArrays));

                Expr ThreadsInSameGroup_BothEnabled_AtLeastOneGlobalFence = 
                  Expr.And(Expr.And(GPUVerifier.ThreadsInSameGroup(), Expr.And(P1, P2)), Expr.Or(GlobalFence1, GlobalFence2));

                if (SomeArrayModelledNonAdversarially(GlobalArrays)) {
                  var NoAccessVars = GPUVerifyVCGenCommandLineOptions.BarrierAccessChecks ? 
                    GlobalArrays.Select(x => FindOrCreateNotAccessedVariable(x.Name, (x.TypedIdent.Type as MapType).Arguments[0])) :
                    Enumerable.Empty<Variable>();
                  var HavocVars = GlobalArrays.Concat(NoAccessVars).ToList();
                  bigblocks.Add(new BigBlock(Token.NoToken, null, new List<Cmd>(),
                    new IfCmd(Token.NoToken,
                      ThreadsInSameGroup_BothEnabled_AtLeastOneGlobalFence,
                      new StmtList(MakeHavocBlocks(HavocVars), Token.NoToken), null, null), null));
                }
            }

            if(RaceInstrumentationUtil.RaceCheckingMethod != RaceCheckingMethod.STANDARD) {
              bigblocks.Add(new BigBlock(Token.NoToken, null, new List<Cmd> {
                new HavocCmd(Token.NoToken, new List<IdentifierExpr> {
                  new IdentifierExpr(Token.NoToken, new GlobalVariable(Token.NoToken,
                    new TypedIdent(Token.NoToken, "_TRACKING", Microsoft.Boogie.Type.Bool)))
                })
              }, null, null));

            }

            StmtList statements = new StmtList(bigblocks, BarrierProcedure.tok);
            Implementation BarrierImplementation = 
                new Implementation(BarrierProcedure.tok, BarrierProcedure.Name, new List<TypeVariable>(), 
                    BarrierProcedure.InParams, BarrierProcedure.OutParams, new List<Variable>(), statements);

            BarrierImplementation.Resolve(ResContext);

            AddInlineAttribute(BarrierImplementation);
            AddInlineAttribute(BarrierProcedure);

            BarrierImplementation.Proc = BarrierProcedure;

            Program.TopLevelDeclarations.Add(BarrierImplementation);
        }

        private NAryExpr MakeFenceExpr(Variable v) {
          return Expr.Neq(new IdentifierExpr(Token.NoToken, new LocalVariable(Token.NoToken, v.TypedIdent)), 
            IntRep.GetLiteral(0, 1));
        }

        private List<BigBlock> MakeResetBlocks(Expr ResetCondition, ICollection<Variable> variables)
        {
            Debug.Assert(variables.Count > 0);
            List<BigBlock> result = new List<BigBlock>();
            foreach (Variable v in variables)
            {
                result.Add(RaceInstrumenter.MakeResetReadWriteSetStatements(v, ResetCondition));
            }
            Debug.Assert(result.Count > 0);
            return result;
        }

        private List<BigBlock> MakeHavocBlocks(ICollection<Variable> variables) {
          Debug.Assert(variables.Count > 0);
          List<BigBlock> result = new List<BigBlock>();
          foreach (Variable v in variables) {
            // Revisit: how to havoc NOT_ACCESSED vars properly
            if (!ArrayModelledAdversarially(v) || v.Name.Contains("_NOT_ACCESSED_")) {
              result.Add(HavocSharedArray(v));
            }
          }
          Debug.Assert(result.Count > 0);
          return result;
        }

        private bool SomeArrayModelledNonAdversarially(ICollection<Variable> variables) {
          foreach (Variable v in variables) {
            if (!ArrayModelledAdversarially(v)) {
              return true;
            }
          }
          return false;
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
            return new BigBlock(Token.NoToken, null, 
              new List<Cmd>(new Cmd[] { new HavocCmd(Token.NoToken, 
                new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(Token.NoToken, v) })) }), null, null);
        }

        internal static bool ModifiesSetContains(List<IdentifierExpr> Modifies, IdentifierExpr v)
        {
          return Modifies.Where(Item => Item.Name.Equals(v.Name)).Count() > 0;
        }

        private void AbstractSharedState()
        {
          new AdversarialAbstraction(this).Abstract();
        }

        internal static string MakeBenignFlagVariableName(string Name) {
          return "_WRITE_READ_BENIGN_FLAG_" + Name;
        }

        internal static GlobalVariable MakeBenignFlagVariable(string name) {
          return new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, MakeBenignFlagVariableName(name),
            Microsoft.Boogie.Type.Bool));
        }

        internal GlobalVariable FindOrCreateNotAccessedVariable(string varName, Microsoft.Boogie.Type dtype)
        {
            string name = MakeNotAccessedVariableName(varName);
            foreach(Declaration D in Program.TopLevelDeclarations)
            {
                if(D is GlobalVariable && ((GlobalVariable)D).Name.Equals(name))
                {
                    return D as GlobalVariable;
                }
            }

            GlobalVariable result = MakeNotAccessedVariable(varName, dtype);

            Program.TopLevelDeclarations.Add(result);
            return result;
        }

        internal GlobalVariable FindOrCreateAccessHasOccurredVariable(string varName, AccessType Access)
        {
            foreach(var g in Program.TopLevelDeclarations.OfType<GlobalVariable>()) {
              if(g.Name.Equals(RaceInstrumentationUtil.MakeHasOccurredVariableName(varName, Access))) {
                return g;
              }
            }
            GlobalVariable result = MakeAccessHasOccurredVariable(varName, Access);
            Program.TopLevelDeclarations.Add(result);
            return result;
        }

        internal Variable FindOrCreateOffsetVariable(string varName, AccessType Access)
        {
            foreach(var g in Program.TopLevelDeclarations.OfType<Variable>()) {
              if(g.Name.Equals(RaceInstrumentationUtil.MakeOffsetVariableName(varName, Access))) {
                return g;
              }
            }
            Variable result = RaceInstrumentationUtil.MakeOffsetVariable(varName, Access, IntRep.GetIntType(size_t_bits));
            Program.TopLevelDeclarations.Add(result);
            return result;
        }

        internal Variable FindOrCreateValueVariable(string varName, AccessType Access,
              Microsoft.Boogie.Type Type) {
          foreach(var g in Program.TopLevelDeclarations.OfType<Variable>()) {
            if(g.Name.Equals(RaceInstrumentationUtil.MakeValueVariableName(varName, Access))) {
              return g;
            }
          }
          Variable result = RaceInstrumentationUtil.MakeValueVariable(varName, Access, Type);
          Program.TopLevelDeclarations.Add(result);
          return result;
        }

        internal GlobalVariable FindOrCreateBenignFlagVariable(string varName)
        {
          foreach(var g in Program.TopLevelDeclarations.OfType<GlobalVariable>()) {
            if(g.Name.Equals(MakeBenignFlagVariableName(varName))) {
              return g;
            }
          }
          GlobalVariable result = MakeBenignFlagVariable(varName);
          Program.TopLevelDeclarations.Add(result);
          return result;
        }

        internal Variable FindOrCreateAsyncHandleVariable(string varName, AccessType Access)
        {
            foreach(var g in Program.TopLevelDeclarations.OfType<Variable>()) {
              if(g.Name.Equals(RaceInstrumentationUtil.MakeAsyncHandleVariableName(varName, Access))) {
                return g;
              }
            }
            Variable result = RaceInstrumentationUtil.MakeAsyncHandleVariable(varName, Access, IntRep.GetIntType(size_t_bits));
            Program.TopLevelDeclarations.Add(result);
            return result;
        }


        internal static GlobalVariable MakeNotAccessedVariable(string varName, Microsoft.Boogie.Type dtype)
        {
            var v = new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, MakeNotAccessedVariableName(varName), dtype));
            v.Attributes = new QKeyValue(Token.NoToken, "check_access", new List<object>(new object[] { }), null);
            return v;
        }

        internal static string MakeNotAccessedVariableName(string varName) {
            return "_NOT_ACCESSED_" + varName;
        }

        internal static GlobalVariable MakeAccessHasOccurredVariable(string varName, AccessType Access)
        {
            return new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, RaceInstrumentationUtil.MakeHasOccurredVariableName(varName, Access), Microsoft.Boogie.Type.Bool));
        }

        internal static IdentifierExpr MakeAccessHasOccurredExpr(string varName, AccessType Access)
        {
            return new IdentifierExpr(Token.NoToken, MakeAccessHasOccurredVariable(varName, Access));
        }

        internal Function FindOrCreateOther(int BvWidth) {
          Function other = (Function)ResContext.LookUpProcedure("__other_bv" + BvWidth);
          if (other == null) {
            List<Variable> myargs = new List<Variable>();
            myargs.Add(new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", Microsoft.Boogie.Type.GetBvType(BvWidth))));
            other = new Function(Token.NoToken, "__other_bv" + BvWidth, myargs,
             new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "", Microsoft.Boogie.Type.GetBvType(BvWidth))));
          }
          return other;
        }

        private void MakeKernelDualised()
        {
          new KernelDualiser(this).DualiseKernel();
        }

        private void MakeKernelPredicated()
        {
            if (GPUVerifyVCGenCommandLineOptions.SmartPredication)
            {
                SmartBlockPredicator.Predicate(Program, proc => true, uniformityAnalyser);
            }
            else
            {
                BlockPredicator.Predicate(Program, /*createCandidateInvariants=*/GPUVerifyVCGenCommandLineOptions.Inference);
            }
            return;
        }

        private int Check()
        {
            var BarrierProcedure = FindOrCreateBarrierProcedure();

            BarrierProcedures.Add(BarrierProcedure);

            if (ErrorCount > 0)
            {
                return ErrorCount;
            }

            if (BarrierProcedure.InParams.Count() != 2)
            {
                Error(BarrierProcedure, "Barrier procedure must take exactly two arguments");
            }
            else if (!BarrierProcedure.InParams[0].TypedIdent.Type.Equals(
              IntRep.GetIntType(1)))
            {
              Error(BarrierProcedure, "First argument to barrier procedure must have type bv1");
            }
            else if (!BarrierProcedure.InParams[1].TypedIdent.Type.Equals(
              IntRep.GetIntType(1))) {
              Error(BarrierProcedure, "Second argument to barrier procedure must have type bv1");
            }

            if (BarrierProcedure.OutParams.Count() != 0)
            {
                Error(BarrierProcedure, "Barrier procedure must not return any results");
            }

            if(BarrierProcedure.InParams.Count() == 2) {
              BarrierProcedureLocalFenceArgName = BarrierProcedure.InParams[0].Name;
              BarrierProcedureGlobalFenceArgName = BarrierProcedure.InParams[1].Name;
            }

            KernelProcedures = GetKernelProcedures();

            if (!FindNonLocalVariables())
            {
                return ErrorCount;
            }

            return ErrorCount;
        }

        public static bool IsThreadLocalIdConstant(Variable variable)
        {
            return variable.Name.Equals(_X.Name) || variable.Name.Equals(_Y.Name) || variable.Name.Equals(_Z.Name);
        }

        public static bool IsGroupIdConstant(Variable variable)
        {
            return variable.Name.Equals(_GROUP_X.Name) || variable.Name.Equals(_GROUP_Y.Name) || variable.Name.Equals(_GROUP_Z.Name);
        }

        public static bool IsDualisedGroupIdConstant(Variable variable)
        {
            var name = GVUtil.StripThreadIdentifier(variable.Name);
            return name.Equals(_GROUP_X.Name) || name.Equals(_GROUP_Y.Name) || name.Equals(_GROUP_Z.Name);
        }

        internal void AddCandidateInvariant(IRegion region, Expr e, string tag, int StageId, string attribute = null)
        {
            if (GPUVerifyVCGenCommandLineOptions.DoNotGenerateCandidates.Contains(tag)) {
                return; // candidate *not* generated
            }

            PredicateCmd predicate = Program.CreateCandidateInvariant(e, tag, StageId);

            if (attribute != null)
              predicate.Attributes = new QKeyValue(Token.NoToken, attribute, new List<object>() { }, predicate.Attributes);

            region.AddInvariant(predicate);
        }

        internal Implementation GetImplementation(string procedureName)
        {
          var Relevant = Program.Implementations().Where(Item => Item.Name == procedureName);
          return Relevant.Count() == 0 ? null : Relevant.ToList()[0];
        }

        internal Procedure GetProcedure(string procedureName) {
          var Relevant = Program.TopLevelDeclarations.OfType<Procedure>().Where(Item => Item.Name == procedureName);
          Debug.Assert(Relevant.Count() > 0);
          return Relevant.ToList()[0];
        }

        internal bool ContainsBarrierCall(IRegion loop)
        {
          return loop.Cmds().OfType<CallCmd>().Where(Item => IsBarrier(Item.Proc)).Count() > 0;
        }

        internal bool ContainsUnsafeBarrierCall(IRegion loop)
        {
          return loop.Cmds().OfType<CallCmd>().Where(Item => IsBarrier(Item.Proc)
            && !QKeyValue.FindBoolAttribute(Item.Proc.Attributes, "safe_barrier")).Count() > 0;
        }

        internal static bool IsBarrier(Procedure Proc)
        {
          return QKeyValue.FindBoolAttribute(Proc.Attributes, "barrier");
        }

        internal bool ArrayModelledAdversarially(Variable v)
        {
            if (GPUVerifyVCGenCommandLineOptions.AdversarialAbstraction)
            {
                return true;
            }
            if (GPUVerifyVCGenCommandLineOptions.EqualityAbstraction)
            {
                return false;
            }
            return !arrayControlFlowAnalyser.MayAffectControlFlow(v.Name);
        }

        internal Expr GlobalIdExpr(string dimension)
        {
            return IntRep.MakeAdd(IntRep.MakeMul(
                            new IdentifierExpr(Token.NoToken, GetGroupId(dimension)), new IdentifierExpr(Token.NoToken, GetGroupSize(dimension))),
                                new IdentifierExpr(Token.NoToken, MakeThreadId(dimension)));
        }

        internal IRegion RootRegion(Implementation Impl)
        {
          return new UnstructuredRegion(Program, Impl);
        }


        public static bool IsGivenConstant(Expr e, Constant c)
        {
            if (!(e is IdentifierExpr))
                return false;

            var varName = ((IdentifierExpr)e).Decl.Name;
            return (GVUtil.StripThreadIdentifier(varName) == GVUtil.StripThreadIdentifier(c.Name));
        }

        public bool SubstIsGivenConstant(Implementation impl, Expr e, Constant c)
        {
            if (!(e is IdentifierExpr))
                return false;
            e = varDefAnalyses[impl].SubstDefinitions(e, impl.Name);
            return IsGivenConstant(e, c);
        }

        public Constant GetLocalIdConst(int dim)
        {
            switch (dim)
            {
                case 0:  return _X;
                case 1:  return _Y;
                case 2:  return _Z;
                default: Debug.Assert(false);
                         return null;
            }
        }

        public Constant GetGroupIdConst(int dim)
        {
            switch (dim)
            {
                case 0:  return _GROUP_X;
                case 1:  return _GROUP_Y;
                case 2:  return _GROUP_Z;
                default: Debug.Assert(false);
                         return null;
            }
        }

        public Constant GetGroupSizeConst(int dim)
        {
            switch (dim)
            {
                case 0:  return _GROUP_SIZE_X;
                case 1:  return _GROUP_SIZE_Y;
                case 2:  return _GROUP_SIZE_Z;
                default: Debug.Assert(false);
                         return null;
            }
        }

        public bool IsLocalId(Expr e, int dim, Implementation impl)
        {
            return SubstIsGivenConstant(impl, e, GetLocalIdConst(dim));
        }

        public bool IsGlobalId(Expr e, int dim, Implementation impl)
        {
            e = varDefAnalyses[impl].SubstDefinitions(e, impl.Name);

            if (e is NAryExpr && (e as NAryExpr).Fun.FunctionName.Equals("BV" + size_t_bits + "_ADD"))
            {
                NAryExpr nary = e as NAryExpr;
                Constant localId = GetLocalIdConst(dim);

                if (IsGivenConstant(nary.Args[1], localId))
                {
                    return IsGroupIdTimesGroupSize(nary.Args[0], dim);
                }

                if (IsGivenConstant(nary.Args[0], localId))
                {
                    return IsGroupIdTimesGroupSize(nary.Args[1], dim);
                }
            }

            return false;
        }

        private bool IsGroupIdTimesGroupSize(Expr expr, int dim)
        {
            if (expr is NAryExpr && (expr as NAryExpr).Fun.FunctionName.Equals("BV" + size_t_bits + "_MUL"))
            {
                NAryExpr innerNary = expr as NAryExpr;

                if (IsGroupIdAndSize(dim, innerNary.Args[0], innerNary.Args[1]))
                {
                    return true;
                }

                if (IsGroupIdAndSize(dim, innerNary.Args[1], innerNary.Args[0]))
                {
                    return true;
                }
            }
            return false;
        }

        private bool IsGroupIdAndSize(int dim, Expr maybeGroupId, Expr maybeGroupSize)
        {
            return IsGivenConstant(maybeGroupId, GetGroupIdConst(dim)) &&
                   IsGivenConstant(maybeGroupSize, GetGroupSizeConst(dim));
        }

        internal Expr MaybeDualise(Expr e, int id, string procName)
        {
            if (id == 0 || e == null)
                return e;
            else
                return (Expr)new VariableDualiser(id, uniformityAnalyser, procName).Visit(e.Clone());
        }

        internal static bool IsConstantInCurrentRegion(IdentifierExpr expr) {
          return (expr.Decl is Constant) ||
                 (expr.Decl is Formal && ((Formal)expr.Decl).InComing);
        }

        internal static Expr GroupSharedIndexingExpr(int Thread) {
          return Thread == 1 ? Expr.True : ThreadsInSameGroup();
        }

        internal bool ProgramUsesBarrierInvariants() {
          foreach (var b in Program.Blocks()) {
            foreach (Cmd c in b.Cmds) {
              if (c is CallCmd) {
                if (QKeyValue.FindBoolAttribute((c as CallCmd).Proc.Attributes, "barrier_invariant")) {
                  return true;
                }
              }
            }
          }
          return false;
        }


        internal static void AddInlineAttribute(Declaration d)
        {
          d.AddAttribute("inline", new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(1)) });
        }

        // This finds instances where the only atomic used on an array satisfy forall n,m f^n(x) != f^m(x)
        // Technically unsound due to machine integers, but unlikely in practice due to, e.g., needing to callng atomic_inc >2^32 times
        private void RefineAtomicAbstraction()
        {
          var implementations = Program.TopLevelDeclarations.Where(d => d is Implementation).Select(d => d as Implementation);
          var blocks = implementations.SelectMany(impl => impl.Blocks);

          // First, pass over the the program looking for uses of atomics, recording (Array,Function) pairs
          Dictionary<Variable,HashSet<string>> funcs_used = new Dictionary<Variable,HashSet<string>> ();
          Dictionary<Variable,HashSet<Expr>> args_used = new Dictionary<Variable,HashSet<Expr>> ();
          foreach (Block b in blocks)
            foreach (Cmd c in b.Cmds)
              if (c is CallCmd)
              {
                CallCmd call = c as CallCmd;
                if (QKeyValue.FindBoolAttribute(call.Attributes, "atomic"))
                {
                  Variable v = (call.Ins[0] as IdentifierExpr).Decl;
                  if (funcs_used.ContainsKey(v))
                    funcs_used[v].Add(QKeyValue.FindStringAttribute(call.Attributes, "atomic_function"));
                  else
                    funcs_used.Add(v, new HashSet<string> (new string[] { QKeyValue.FindStringAttribute(call.Attributes, "atomic_function") }));
                  Expr arg = QKeyValue.FindExprAttribute(call.Attributes, "arg1");
                  if (arg != null)
                  {
                    if (args_used.ContainsKey(v))
                      args_used[v].Add(arg);
                    else
                      args_used.Add(v, new HashSet<Expr> (new Expr[] { arg }));
                  }
                }
              }
          // Then, for every array that only used a single monotonic atomic function, pass over the program again, logging offset constraints
          string[] monotonics = new string[] { "__atomic_inc", "__atomic_dec", "__atomic_add", "__atomic_sub", "__atomicAdd", "__atomicSub" };
          Expr variables = null;
          Expr offset = null;
          int parts = 0;
          foreach (KeyValuePair<Variable,HashSet<string>> pair in funcs_used)
          {
            // If it's a refinable function, and either has no arguments (is inc or dec), or has only 1 argument used with it and that argument is a non-zero constant
            if (pair.Value.Count == 1 && monotonics.Any(x => pair.Value.ToArray()[0].StartsWith(x)) && (!args_used.ContainsKey(pair.Key) || (args_used[pair.Key].Count == 1 &&
                  args_used[pair.Key].All(arg => (arg is LiteralExpr) && ((arg as LiteralExpr).Val is BvConst) && ((arg as LiteralExpr).Val as BvConst).Value != BigNum.FromInt(0)))))
            {
              foreach (Block b in blocks)
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
                      else {
                        variables = new BvConcatExpr(Token.NoToken,call.Outs[0], variables);
                        variables.Type = variables.ShallowType;
                      }
                      if (QKeyValue.FindIntAttribute(call.Attributes, "part", -1) == parts)
                      {
                        AssumeCmd assume = new AssumeCmd(Token.NoToken, Expr.True);
                        assume.Attributes = new QKeyValue(Token.NoToken, "atomic_refinement", new List<object>(new object [] {}), null);
                        assume.Attributes = new QKeyValue(Token.NoToken, "variable", new List<object>(new object[] {variables}), assume.Attributes);
                        assume.Attributes = new QKeyValue(Token.NoToken, "offset", new List<object>(new object[] {offset}), assume.Attributes);
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

        internal GlobalVariable FindOrCreateUsedMap(string arrayName, Microsoft.Boogie.Type elementType)
        {
          string name = "_USED_" + arrayName;

          var CandidateVariables = Program.TopLevelDeclarations.OfType<GlobalVariable>().Where(Item => Item.Name.Equals(name));
          if(CandidateVariables.Count() > 0) {
            return CandidateVariables.ToList()[0];
          }

          Microsoft.Boogie.Type mapType = new MapType(
            Token.NoToken,
            new List<TypeVariable>(), 
            new List<Microsoft.Boogie.Type> { IntRep.GetIntType(size_t_bits) },
            new MapType(Token.NoToken,
                        new List<TypeVariable>(),
                        new List<Microsoft.Boogie.Type> { elementType },
                        Microsoft.Boogie.Type.Bool));
          GlobalVariable usedMap = new GlobalVariable(Token.NoToken, new TypedIdent(Token.NoToken, name, mapType));
          usedMap.Attributes = new QKeyValue(Token.NoToken, "atomic_usedmap", new List<object>(new object [] {}), null);
          Program.TopLevelDeclarations.Add(usedMap);
          return usedMap;
        }

        private Expr FlattenedThreadId(int thread)
        {
          return IntRep.MakeAdd(Expr.Ident(MakeThreadId("X",thread)), IntRep.MakeAdd(
                IntRep.MakeMul(Expr.Ident(MakeThreadId("Y",thread)), Expr.Ident(GetGroupSize("X"))), 
                IntRep.MakeMul(Expr.Ident(MakeThreadId("Z",thread)), IntRep.MakeMul(Expr.Ident(GetGroupSize("X")),Expr.Ident(GetGroupSize("Y"))))));
        }

        private void DoOnlyWarp()
        {
          Expr condition = Expr.And(ThreadsInSameGroup(), ThreadsInSameWarp());
          Program.TopLevelDeclarations.Add(new Axiom(Token.NoToken, condition));

          foreach (Implementation impl in Program.TopLevelDeclarations.OfType<Implementation>())
          {
            impl.Blocks = impl.Blocks.Select(WarpResets).ToList();
          }
        }

        private Block WarpResets(Block b)
        {
          var result = new List<Cmd>();
          foreach (Cmd c in b.Cmds)
          {
            result.Add(c);
            if (c is CallCmd)
            {
              CallCmd call = c as CallCmd;
              if (call.callee.StartsWith("_CHECK_WRITE"))
              {
                foreach (Variable v in KernelArrayInfo.getAllNonLocalArrays())
                {
                  if (v.Name.Equals(call.callee.Substring(13)) && (!ArrayModelledAdversarially(v) || v.Name.Contains("_NOT_ACCESSED")))
                    result.Add(new HavocCmd(Token.NoToken, new List<IdentifierExpr>(new IdentifierExpr[] {new IdentifierExpr(Token.NoToken,v)})));
                }
              }
            }
          }
          b.Cmds = result;
          return b;
        }

        private void DoNoWarp()
        {
          Expr condition = Expr.Imp(ThreadsInSameGroup(), Expr.Not(ThreadsInSameWarp()));
          Program.TopLevelDeclarations.Add(new Axiom(Token.NoToken, condition));
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

          foreach (Procedure proto in WarpSyncs.Values) {
            AddInlineAttribute(proto);
            Program.TopLevelDeclarations.Add(proto);
          }
        }

        private Block AddWarpSyncs(Block b)
        {
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
                if (call.callee.StartsWith("_LOG_ATOMIC")) {
                  kind = AccessType.ATOMIC;
                  array = call.callee.Substring(12); // "_LOG_ATOMIC_" is 14 characters
                } else if (call.callee.StartsWith("_LOG_READ")) {
                  kind = AccessType.READ;
                  array = call.callee.Substring(10); // "_LOG_READ_" is 12 characters
                } else {
                  Debug.Assert(call.callee.StartsWith("_LOG_WRITE"));
                  kind = AccessType.WRITE;
                  array = call.callee.Substring(11); // "_LOG_WRITE_" is 13 characters
                }
                // Manual resolving yaey!
                Variable arrayVar = KernelArrayInfo.getAllArrays().Where(v => v.Name.Equals(array)).First();
                Procedure proto = FindOrCreateWarpSync(arrayVar,kind,true);
                CallCmd wsCall = new CallCmd(Token.NoToken,proto.Name,new List<Expr>(),new List<IdentifierExpr>());
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
                if (call.callee.StartsWith("_CHECK_ATOMIC")) {
                  kind = AccessType.ATOMIC;
                  array = call.callee.Substring(14); // "_CHECK_ATOMIC_" is 14 characters
                } else if (call.callee.StartsWith("_CHECK_READ")) {
                  kind = AccessType.READ;
                  array = call.callee.Substring(12); // "_CHECK_READ_" is 12 characters
                } else {
                  Debug.Assert(call.callee.StartsWith("_CHECK_WRITE"));
                  kind = AccessType.WRITE;
                  array = call.callee.Substring(13); // "_CHECK_WRITE_" is 13 characters
                }
                // Manual resolving yaey!
                Variable arrayVar = KernelArrayInfo.getAllArrays().Where(v => v.Name.Equals(array)).First();
                Procedure proto = FindOrCreateWarpSync(arrayVar,kind,false);
                CallCmd wsCall = new CallCmd(Token.NoToken,proto.Name,new List<Expr>(),new List<IdentifierExpr>());
                wsCall.Proc = proto;
                result.Add(wsCall);
              }
            }
          }
          b.Cmds = result;
          return b;
        }

        private Procedure FindOrCreateWarpSync(Variable array, AccessType kind, bool pre) {
          Tuple<Variable,AccessType,bool> key = new Tuple<Variable,AccessType,bool>(array,kind,pre);
          if (!WarpSyncs.ContainsKey(key)) {
            Procedure proto = new Procedure (Token.NoToken, (pre?"_PRE":"_POST") + "_WARP_SYNC_" + array.Name + "_" + kind, new List<TypeVariable>(), new List<Variable>(), new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(), new List<Ensures>());
            WarpSyncs[key] = proto;
          }
          return WarpSyncs[key];
        }

        private void GenerateWarpSyncs()
        {
          foreach (Tuple<Variable,AccessType,bool> pair in WarpSyncs.Keys)
          {
            Variable v = pair.Item1;
            AccessType kind = pair.Item2;
            bool pre = pair.Item3;
            Procedure SyncProcedure = WarpSyncs[pair];

            Expr P1 = null;
            Expr P2 = null;

            if (uniformityAnalyser.IsUniform(SyncProcedure.Name)) {
              P1 = Expr.True;
              P2 = Expr.True;
            }
            else { // If not uniform, should be predicated -- we don't take any other parameters...
              P1 = Expr.Ident(SyncProcedure.InParams[0]);
              P2 = Expr.Ident(SyncProcedure.InParams[1]);
            }

            // Implementation
            List<Cmd> then = new List<Cmd>();
            if (pre) {
              var reset_needs = new[] { new {Kind = AccessType.READ,   Resets = new[] {AccessType.WRITE, AccessType.ATOMIC}}
                                      , new {Kind = AccessType.WRITE,  Resets = new[] {AccessType.READ, AccessType.WRITE, AccessType.ATOMIC}}
                                      , new {Kind = AccessType.ATOMIC, Resets = new[] {AccessType.READ, AccessType.WRITE}}
              };
              foreach (AccessType a in reset_needs.Where(x => x.Kind == kind).First().Resets) {
                Variable accessVariable = FindOrCreateAccessHasOccurredVariable(v.Name,a);
                then.Add(new AssumeCmd (Token.NoToken, Expr.Not(Expr.Ident(accessVariable))));
              }
            }
            else {
              Variable accessVariable = FindOrCreateAccessHasOccurredVariable(v.Name,kind);
              then.Add(new AssumeCmd (Token.NoToken, Expr.Not(Expr.Ident(accessVariable))));
            }
            List<BigBlock> thenblocks = new List<BigBlock>();
            thenblocks.Add(new BigBlock(Token.NoToken, "reset_warps", then, null, null));

            if (kind == AccessType.WRITE && !ArrayModelledAdversarially(v)) {
              thenblocks.AddRange(MakeHavocBlocks(new Variable[] {v}));
            }

            Expr condition = Expr.And(Expr.Eq(P1,P2),Expr.And(ThreadsInSameGroup(), ThreadsInSameWarp()));

            IfCmd ifcmd = new IfCmd (Token.NoToken, condition, new StmtList (thenblocks,Token.NoToken), /* another IfCmd for elsif */ null, /* then branch */ null);

            List<BigBlock> blocks = new List<BigBlock>();
            blocks.Add(new BigBlock(Token.NoToken,"entry", new List<Cmd>(),ifcmd ,null));

            Implementation method = new Implementation(Token.NoToken, SyncProcedure.Name, new List<TypeVariable>(), SyncProcedure.InParams, new List<Variable>(), new List<Variable>(), new StmtList(blocks,Token.NoToken));
            AddInlineAttribute(method);
            Program.TopLevelDeclarations.Add(method);
          }
        }

        internal void AddRegionWithLoopInvariantsDisabled(IRegion region) {
            RegionsWithLoopInvariantsDisabled.Add(region.Identifier());
        }

        internal bool RegionHasLoopInvariantsDisabled(IRegion region) {
            return RegionsWithLoopInvariantsDisabled.Contains(region.Identifier());
        }

        private void AddLoopInvariantDisabledTags() {
            foreach (var impl in Program.Implementations().ToList())
            {
                foreach (var region in RootRegion(impl).SubRegions())
                {
                    if (RegionHasLoopInvariantsDisabled(region))
                        region.AddLoopInvariantDisabledTag();
                }
            }
        }

        internal bool TryGetArrayFromPrefixedString(string s, string prefix, out Variable v) {
          v = null;
          if(s.StartsWith(prefix)) {
            foreach(var a in KernelArrayInfo.getAllNonLocalArrays()) {
              if(a.Name.Equals(s.Substring(prefix.Length))) {
                v = a;
                return true;
              }
            }
          }
          return false;
        }

        internal bool TryGetArrayFromAccessHasOccurred(string s, AccessType Access, out Variable v) {
          return TryGetArrayFromPrefixedString(s, "_" + Access + "_HAS_OCCURRED_", out v);
        }

        internal bool TryGetArrayFromLogOrCheckProcedure(string s, AccessType Access, string logOrCheck, out Variable v)
        {
          return TryGetArrayFromPrefixedString(s, "_" + logOrCheck + "_" + Access + "_", out v);
        }

        internal bool TryGetArrayFromLogProcedure(string s, AccessType Access, out Variable v)
        {
          return TryGetArrayFromLogOrCheckProcedure(s, Access, "LOG", out v);
        }

        internal bool TryGetArrayFromCheckProcedure(string s, AccessType Access, out Variable v)
        {
          return TryGetArrayFromLogOrCheckProcedure(s, Access, "CHECK", out v);
        }

        internal Variable FindOrCreateEnabledVariable() {
          string enabledVariableName = "__enabled";
          Variable enabledVariable = (Variable)ResContext.LookUpVariable(enabledVariableName);
          if (enabledVariable == null) {
            enabledVariable = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, enabledVariableName, Microsoft.Boogie.Type.Bool), false);
            enabledVariable.AddAttribute("__enabled");
            ResContext.AddVariable(enabledVariable, true);
          }
          return enabledVariable;
        }

        internal Expr FindOrCreateAsyncNoHandleConstant() {
          string Name = "_ASYNC_NO_HANDLE";
          var Candidates = Program.TopLevelDeclarations.OfType<Constant>().Where(Item => Item.Name == Name);
          if(Candidates.Count() > 0) {
            Debug.Assert(Candidates.Count() == 1);
            return Expr.Ident(Candidates.ToList()[0]);
          }
          Constant AsyncNoHandleConstant = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, Name, IntRep.GetIntType(size_t_bits)), false);
          Axiom EqualsZero = new Axiom(Token.NoToken, Expr.Eq(Expr.Ident(AsyncNoHandleConstant), Zero(size_t_bits)));
          Program.TopLevelDeclarations.AddRange(new Declaration[] { AsyncNoHandleConstant, EqualsZero });
          return Expr.Ident(AsyncNoHandleConstant);
        }

    }
}
