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
    using System.IO;
    using System.Linq;
    using System.Numerics;
    using Microsoft.Boogie;
    using Microsoft.Boogie.GraphUtil;

    public class GPUVerifyErrorReporter
    {
        private enum ErrorMsgType
        {
            Error,
            Note,
            NoError
        }

        private static void ErrorWriteLine(string locInfo, string message, ErrorMsgType msgtype)
        {
            Contract.Requires(message != null);
            ConsoleColor col = Console.ForegroundColor;
            if (!string.IsNullOrEmpty(locInfo))
            {
                Console.Error.Write(locInfo + " ");
            }

            switch (msgtype)
            {
                case ErrorMsgType.Error:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Error.Write("error: ");
                    break;
                case ErrorMsgType.Note:
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.Error.Write("note: ");
                    break;
                case ErrorMsgType.NoError:
                default:
                    break;
            }

            Console.ForegroundColor = col;
            Console.Error.WriteLine(message);
        }

        private const string SizeTBitsType = "_SIZE_T_TYPE";
        private readonly int sizeTBits;
        private readonly Dictionary<string, string> globalArraySourceNames;
        private Implementation impl;

        public GPUVerifyErrorReporter(Program program, string implName)
        {
            impl = program.Implementations.Where(item => item.Name.Equals(implName)).First();
            sizeTBits = GetSizeTBits(program);

            globalArraySourceNames = new Dictionary<string, string>();
            foreach (var g in program.TopLevelDeclarations.OfType<GlobalVariable>())
            {
                string sourceName = QKeyValue.FindStringAttribute(g.Attributes, "source_name");
                if (sourceName != null)
                    globalArraySourceNames[g.Name] = sourceName;
                else
                    globalArraySourceNames[g.Name] = g.Name;
            }
        }

        private int GetSizeTBits(Program program)
        {
            var candidates = program.TopLevelDeclarations.OfType<TypeSynonymDecl>()
                .Where(item => item.Name == SizeTBitsType);
            if (candidates.Count() != 1 || !candidates.First().Body.IsBv)
            {
                Console.WriteLine("GPUVerify: error: exactly one _SIZE_T_TYPE bit-vector type must be specified");
                Environment.Exit((int)ToolExitCodes.OTHER_ERROR);
            }

            return candidates.First().Body.BvBits;
        }

        public void ReportCounterexample(Counterexample error)
        {
            int windowWidth;
            try
            {
                windowWidth = Console.WindowWidth;
            }
            catch (IOException)
            {
                windowWidth = 20;
            }

            for (int i = 0; i < windowWidth; i++)
                Console.Error.Write("-");

            if (error is CallCounterexample)
            {
                CallCounterexample callCex = (CallCounterexample)error;
                if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "barrier_divergence"))
                    ReportBarrierDivergence(callCex.FailingCall);
                else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "race"))
                    ReportRace(callCex);
                else
                    ReportRequiresFailure(callCex.FailingCall, callCex.FailingRequires);
            }
            else if (error is ReturnCounterexample)
            {
                ReturnCounterexample returnCex = (ReturnCounterexample)error;
                ReportEnsuresFailure(returnCex.FailingEnsures);
            }
            else
            {
                AssertCounterexample assertCex = (AssertCounterexample)error;
                if (assertCex.FailingAssert is LoopInitAssertCmd)
                    ReportInvariantEntryFailure(assertCex);
                else if (assertCex.FailingAssert is LoopInvMaintainedAssertCmd)
                    ReportInvariantMaintedFailure(assertCex);
                else if (QKeyValue.FindBoolAttribute(assertCex.FailingAssert.Attributes, "barrier_invariant"))
                    ReportFailingBarrierInvariant(assertCex);
                else if (QKeyValue.FindBoolAttribute(assertCex.FailingAssert.Attributes, "barrier_invariant_access_check"))
                    ReportFailingBarrierInvariantAccessCheck(assertCex);
                else if (QKeyValue.FindBoolAttribute(assertCex.FailingAssert.Attributes, "constant_write"))
                    ReportFailingConstantWriteCheck(assertCex);
                else if (QKeyValue.FindBoolAttribute(assertCex.FailingAssert.Attributes, "bad_pointer_access"))
                    ReportFailingBadPointerAccess(assertCex);
                else if (QKeyValue.FindBoolAttribute(assertCex.FailingAssert.Attributes, "array_bounds"))
                    ReportFailingArrayBounds(assertCex);
                else
                    ReportFailingAssert(assertCex);
            }

            DisplayParameterValues(error);

            if (((GVCommandLineOptions)CommandLineOptions.Clo).DisplayLoopAbstractions)
            {
                DisplayLoopAbstractions(error);
            }
        }

        private void DisplayLoopAbstractions(Counterexample error)
        {
            PopulateModelWithStatesIfNecessary(error);
            Program originalProgram = GetOriginalProgram();
            var cfg = originalProgram.ProcessLoops(GetOriginalImplementation(originalProgram));

            for (int i = 0; i < error.Trace.Count(); i++)
            {
                MaybeDisplayLoopHeadState(error.Trace[i], cfg, error.Model, originalProgram);
                if (i < error.Trace.Count() - 1)
                {
                    MaybeDisplayLoopEntryState(error.Trace[i], error.Trace[i + 1], cfg, error.Model, originalProgram);
                }

                MaybeDisplayLoopBackEdgeState(error.Trace[i], cfg, error.Model, originalProgram);
            }
        }

        private void MaybeDisplayLoopEntryState(Block current, Block next, Graph<Block> cfg, Model model, Program originalProgram)
        {
            var loopHeadState = FindLoopHeadState(next);
            if (loopHeadState == null)
            {
                return;
            }

            Block header = FindLoopHeaderWithStateName(loopHeadState, cfg);
            var loopHeadStateSuffix = loopHeadState.Substring("loop_head_state_".Count());
            var relevantLoopEntryStates = GetCaptureStates(current).Where(item => item.Contains("loop_entry_state_" + loopHeadStateSuffix));
            if (relevantLoopEntryStates.Count() == 0)
            {
                return;
            }

            Debug.Assert(relevantLoopEntryStates.Count() == 1);
            var loopEntryState = relevantLoopEntryStates.First();
            Console.WriteLine("On entry to loop headed at " + GetSourceLocationForBasicBlock(header).Top() + ":");
            ShowRaceInstrumentationVariables(model, loopEntryState, originalProgram);
            ShowVariablesReferencedInLoop(cfg, model, loopEntryState, header);
        }

        private SourceLocationInfo GetSourceLocationForBasicBlock(Block header)
        {
            foreach (var a in header.Cmds.OfType<AssertCmd>())
            {
                if (QKeyValue.FindBoolAttribute(a.Attributes, "block_sourceloc"))
                {
                    return new SourceLocationInfo(a.Attributes, GetSourceFileName(), header.tok);
                }
            }

            Debug.Assert(false);
            return null;
        }

        private void MaybeDisplayLoopBackEdgeState(Block current, Graph<Block> cfg, Model model, Program originalProgram)
        {
            var relevantLoopBackEdgeStates = GetCaptureStates(current).Where(item => item.Contains("loop_back_edge_state"));
            if (relevantLoopBackEdgeStates.Count() == 0)
            {
                return;
            }

            Debug.Assert(relevantLoopBackEdgeStates.Count() == 1);
            var loopBackEdgeState = relevantLoopBackEdgeStates.First();
            if (GetStateFromModel(loopBackEdgeState, model) == null)
            {
                return;
            }

            var originalHeader = FindHeaderForBackEdgeNode(cfg, FindNodeContainingCaptureState(cfg, loopBackEdgeState));
            Console.WriteLine("On taking back edge to head of loop at " +
              GetSourceLocationForBasicBlock(originalHeader).Top() + ":");
            ShowRaceInstrumentationVariables(model, loopBackEdgeState, originalProgram);
            ShowVariablesReferencedInLoop(cfg, model, loopBackEdgeState, originalHeader);
        }

        private void MaybeDisplayLoopHeadState(Block header, Graph<Block> cfg, Model model, Program originalProgram)
        {
            var stateName = FindLoopHeadState(header);
            if (stateName == null)
            {
                return;
            }

            Block originalHeader = FindLoopHeaderWithStateName(stateName, cfg);
            Debug.Assert(header != null);
            Console.Error.WriteLine("After 0 or more iterations of loop headed at "
              + GetSourceLocationForBasicBlock(originalHeader).Top() + ":");
            ShowRaceInstrumentationVariables(model, stateName, originalProgram);
            ShowVariablesReferencedInLoop(cfg, model, stateName, originalHeader);
        }

        private void ShowRaceInstrumentationVariables(Model model, string capturedState, Program originalProgram)
        {
            foreach (var v in originalProgram.TopLevelDeclarations.OfType<Variable>()
                .Where(item => QKeyValue.FindBoolAttribute(item.Attributes, "race_checking")))
            {
                foreach (var t in AccessType.Types)
                {
                    if (v.Name.StartsWith("_" + t + "_HAS_OCCURRED_"))
                    {
                        string arrayName;
                        AccessType access;
                        GetArrayNameAndAccessTypeFromAccessHasOccurredVariable(v, out arrayName, out access);
                        var accessOffsetVar = originalProgram.TopLevelDeclarations.OfType<Variable>()
                            .Where(item => item.Name == RaceInstrumentationUtil.MakeOffsetVariableName(arrayName, access)).First();
                        if (ExtractVariableValueFromCapturedState(v.Name, capturedState, model) == "true")
                        {
                            if (GetStateFromModel(capturedState, model).TryGet(accessOffsetVar.Name) is Model.Number)
                            {
                                Console.Error.WriteLine("  " + access.ToString().ToLower() + " " + access.Direction() + " "
                                  + ArrayOffsetString(model, capturedState, v, accessOffsetVar, arrayName)
                                  + " (" + ThreadDetails(model, 1, false) + ")");
                            }
                            else
                            {
                                Console.Error.WriteLine("  " + access.ToString().ToLower() + " " + access.Direction() + " " + arrayName.TrimStart(new char[] { '$' })
                                  + " (unknown offset)" + " (" + ThreadDetails(model, 1, false) + ")");
                            }
                        }

                        break;
                    }
                }
            }
        }

        private void ShowVariablesReferencedInLoop(Graph<Block> cfg, Model model, string capturedState, Block heaer)
        {
            foreach (var v in VC.VCGen.VarsReferencedInLoop(cfg, heaer).Select(item => item.Name)
                .Where(item => IsOriginalProgramVariable(item)))
            {
                int id;
                var cleaned = CleanOriginalProgramVariable(v, out id);
                Console.Error.Write("  " + cleaned + " = " + ExtractVariableValueFromCapturedState(v, capturedState, model) + " ");
                Console.Error.WriteLine(id == 1 ? "(" + ThreadDetails(model, 1, false) + ")" :
                                       (id == 2 ? "(" + ThreadDetails(model, 2, false) + ")" :
                                                  "(uniform across threads)"));
            }

            Console.Error.WriteLine();
        }

        private void GetArrayNameAndAccessTypeFromAccessHasOccurredVariable(Variable v, out string arrayName, out AccessType accessType)
        {
            Debug.Assert(Utilities.IsAccessHasOccurredVariable(v));
            foreach (var currentAccessType in AccessType.Types)
            {
                var prefix = "_" + currentAccessType + "_HAS_OCCURRED_";
                if (v.Name.StartsWith(prefix))
                {
                    arrayName = globalArraySourceNames[v.Name.Substring(prefix.Count())];
                    accessType = currentAccessType;
                    return;
                }
            }

            Debug.Assert(false);
            arrayName = null;
            accessType = null;
        }

        private bool IsOriginalProgramVariable(string name)
        {
            // We ignore the following variables:
            // * Variables not starting with "$", these are internal variables.
            // * Variables prefixed with "$arrayId", these are internal pointer names.
            return name.Count() > 0 && name.StartsWith("$") && !name.StartsWith("$arrayId");
        }

        private Block FindNodeContainingCaptureState(Graph<Block> cfg, string captureState)
        {
            foreach (var b in cfg.Nodes)
            {
                foreach (var c in b.Cmds.OfType<AssumeCmd>())
                {
                    if (QKeyValue.FindStringAttribute(c.Attributes, "captureState") == captureState)
                        return b;
                }
            }

            return null;
        }

        private Block FindHeaderForBackEdgeNode(Graph<Block> cfg, Block backEdgeNode)
        {
            foreach (var header in cfg.Headers)
            {
                foreach (var currentBackEdgeNode in cfg.BackEdgeNodes(header))
                {
                    if (backEdgeNode == currentBackEdgeNode)
                        return header;
                }
            }

            return null;
        }

        private string FindLoopHeadState(Block b)
        {
            var relevantLoopHeadStates = GetCaptureStates(b).Where(item => item.Contains("loop_head_state"));
            if (relevantLoopHeadStates.Count() == 0)
                return null;

            Debug.Assert(relevantLoopHeadStates.Count() == 1);
            return relevantLoopHeadStates.First();
        }

        private IEnumerable<string> GetCaptureStates(Block b)
        {
            return b.Cmds.OfType<AssumeCmd>()
                .Select(item => QKeyValue.FindStringAttribute(item.Attributes, "captureState"))
                .Where(item => item != null);
        }

        private void DisplayParameterValues(Counterexample error)
        {
            if (impl.InParams.Count() == 0)
                return;

            string funName = QKeyValue.FindStringAttribute(impl.Attributes, "source_name");
            Debug.Assert(funName != null);

            Console.Error.WriteLine("Bitwise values of parameters of '" + funName + "':");
            PopulateModelWithStatesIfNecessary(error);

            foreach (var p in impl.InParams)
            {
                int id;
                string stripped = CleanOriginalProgramVariable(p.Name, out id);

                if (id == 1 || id == 2)
                {
                    Console.Error.WriteLine(
                        " {0} = {1} ({2})", stripped, ExtractVariableValueFromModel(p.Name, error.Model), ThreadDetails(error.Model, id, false));
                }
                else
                {
                    Console.Error.WriteLine(
                        " {0} = {1}", stripped, ExtractVariableValueFromModel(p.Name, error.Model));
                }
            }

            Console.Error.WriteLine();
        }

        private string CleanOriginalProgramVariable(string name, out int id)
        {
            string strippedName = Utilities.StripThreadIdentifier(name, out id);
            if (globalArraySourceNames.ContainsKey(strippedName))
                return globalArraySourceNames[strippedName];
            else
                return strippedName.TrimStart(new char[] { '$' }).Split(new char[] { '.' })[0];
        }

        private static string ExtractValueFromModelElement(Model.Element element)
        {
            if (element is Model.BitVector)
                return ((Model.BitVector)element).Numeral;
            else if (element is Model.Uninterpreted)
                return "<irrelevant>";
            else if (element == null)
                return "<null>";
            else
                return element.ToString(); // "<unknown>";
        }

        private static string ExtractVariableValueFromCapturedState(string variableName, string stateName, Model model)
        {
            return ExtractValueFromModelElement(GetStateFromModel(stateName, model).TryGet(variableName));
        }

        private static string ExtractVariableValueFromModel(string variableName, Model model)
        {
            var func = model.TryGetFunc(variableName);
            if (func != null)
                return ExtractValueFromModelElement(func.GetConstant());
            else
                return "<unknown>";
        }

        private void ReportRace(CallCounterexample callCex)
        {
            PopulateModelWithStatesIfNecessary(callCex);

            string raceyArrayName = GetArrayName(callCex.FailingRequires);
            Debug.Assert(raceyArrayName != null);

            IEnumerable<SourceLocationInfo> possibleSourcesForFirstAccess =
                GetPossibleSourceLocationsForFirstAccessInRace(callCex, raceyArrayName, GetAccessType(callCex), GetStateName(callCex));
            SourceLocationInfo sourceInfoForSecondAccess =
                new SourceLocationInfo(callCex.FailingCall.Attributes, GetSourceFileName(), callCex.FailingCall.tok);

            string raceName, access1, access2;
            DetermineNatureOfRace(callCex, out raceName, out access1, out access2);
            string raceyArraySourceName = GetArraySourceName(callCex.FailingRequires);
            Debug.Assert(raceyArraySourceName != null);

            ErrorWriteLine(
                "\n" + sourceInfoForSecondAccess.Top().GetFile() + ":",
                "possible " + raceName + " race on " + ArrayOffsetString(callCex, raceyArraySourceName) + ":\n",
                ErrorMsgType.Error);

            Console.Error.WriteLine(access2 + " by " + ThreadDetails(callCex.Model, 2, true) + ", " + sourceInfoForSecondAccess.Top() + ":");
            sourceInfoForSecondAccess.PrintStackTrace();

            Console.Error.Write(access1 + " by " + ThreadDetails(callCex.Model, 1, true) + ", ");
            if (possibleSourcesForFirstAccess.Count() == 1)
            {
                Console.Error.WriteLine(possibleSourcesForFirstAccess.First().Top() + ":");
                possibleSourcesForFirstAccess.First().PrintStackTrace();
            }
            else if (possibleSourcesForFirstAccess.Count() == 0)
            {
                Console.Error.WriteLine("from external source location\n");
            }
            else
            {
                Console.Error.WriteLine("possible sources are:");
                List<SourceLocationInfo> locationsAsList = possibleSourcesForFirstAccess.ToList();
                locationsAsList.Sort(new SourceLocationInfo.SourceLocationInfoComparison());
                foreach (var sli in locationsAsList)
                {
                    Console.Error.WriteLine(sli.Top() + ":");
                    sli.PrintStackTrace();
                }

                Console.Error.WriteLine();
            }
        }

        private string ArrayOffsetString(CallCounterexample cex, string raceyArraySourceName)
        {
            Variable accessOffsetVar = ExtractOffsetVar(cex);
            Variable accessHasOccurredVar = ExtractAccessHasOccurredVar(cex);
            string stateName = GetStateName(cex);
            Model model = cex.Model;

            return ArrayOffsetString(model, stateName, accessHasOccurredVar, accessOffsetVar, raceyArraySourceName);
        }

        private string ArrayOffsetString(Model model, string stateName, Variable accessHasOccurredVar, Variable accessOffsetVar, string raceyArraySourceName)
        {
            Model.Element offsetElement = RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.ORIGINAL
              ? GetStateFromModel(stateName, model).TryGet(accessOffsetVar.Name)
              : model.TryGetFunc(accessOffsetVar.Name).GetConstant();

            return GetArrayAccess(
                ParseOffset(offsetElement),
                raceyArraySourceName,
                Convert.ToUInt32(QKeyValue.FindIntAttribute(accessHasOccurredVar.Attributes, "elem_width", -1)),
                Convert.ToUInt32(QKeyValue.FindIntAttribute(accessHasOccurredVar.Attributes, "source_elem_width", -1)),
                QKeyValue.FindStringAttribute(accessHasOccurredVar.Attributes, "source_dimensions").Split(','));
        }

        private static string GetStateName(QKeyValue attributes, Counterexample cex)
        {
            Contract.Requires(QKeyValue.FindStringAttribute(attributes, "check_id") != null);
            string checkId = QKeyValue.FindStringAttribute(attributes, "check_id");
            return QKeyValue.FindStringAttribute(
                cex.Trace.Last().Cmds.OfType<AssumeCmd>()
                    .Where(item => QKeyValue.FindStringAttribute(item.Attributes, "check_id") == checkId)
                    .First().Attributes,
                "captureState");
        }

        protected static string GetStateName(CallCounterexample callCex)
        {
            return GetStateName(callCex.FailingCall.Attributes, callCex);
        }

        private static string GetStateName(AssertCounterexample assertCex)
        {
            return GetStateName(assertCex.FailingAssert.Attributes, assertCex);
        }

        private static string GetSourceFileName()
        {
            return CommandLineOptions.Clo.Files[CommandLineOptions.Clo.Files.Count() - 1];
        }

        protected static void PopulateModelWithStatesIfNecessary(Counterexample cex)
        {
            if (!cex.ModelHasStatesAlready)
            {
                cex.PopulateModelWithStates();
                cex.ModelHasStatesAlready = true;
            }
        }

        private static void DetermineNatureOfRace(CallCounterexample callCex, out string raceName, out string access1, out string access2)
        {
            if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "write_read"))
            {
                raceName = "write-read";
                access1 = "Write";
                access2 = "Read";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "read_write"))
            {
                raceName = "read-write";
                access1 = "Read";
                access2 = "Write";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "write_write"))
            {
                raceName = "write-write";
                access1 = "Write";
                access2 = "Write";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "atomic_read"))
            {
                raceName = "atomic-read";
                access1 = "Atomic";
                access2 = "Read";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "atomic_write"))
            {
                raceName = "atomic-write";
                access1 = "Atomic";
                access2 = "Write";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "read_atomic"))
            {
                raceName = "read-atomic";
                access1 = "Read";
                access2 = "Atomic";
            }
            else if (QKeyValue.FindBoolAttribute(callCex.FailingRequires.Attributes, "write_atomic"))
            {
                raceName = "write-atomic";
                access1 = "Write";
                access2 = "Atomic";
            }
            else
            {
                Debug.Assert(false);
                raceName = null;
                access1 = null;
                access2 = null;
            }
        }

        protected IEnumerable<SourceLocationInfo> GetPossibleSourceLocationsForFirstAccessInRace(CallCounterexample callCex, string arrayName, AccessType accessType, string raceyState)
        {
            string accessHasOccurred = RaceInstrumentationUtil.MakeHasOccurredVariableName(arrayName, accessType);
            string accessOffset = RaceInstrumentationUtil.MakeOffsetVariableName(arrayName, accessType);

            AssumeCmd conflictingAction = DetermineConflictingAction(callCex, raceyState, accessHasOccurred, accessOffset);

            var conflictingState = QKeyValue.FindStringAttribute(conflictingAction.Attributes, "captureState");

            if (conflictingState.Contains("loop_head_state"))
            {
                // The state may have been renamed (for example, if k-induction has been employed),
                // so we need to find the original state name.  This can be computed as the substring before the first
                // occurrence of '$'.  This inversion is fragile, and would be a good candidate for making robust
                string conflictingStatePrefix;
                if (conflictingState.Contains('$'))
                    conflictingStatePrefix = conflictingState.Substring(0, conflictingState.IndexOf('$'));
                else
                    conflictingStatePrefix = conflictingState;

                Program originalProgram = GetOriginalProgram();
                var blockGraph = originalProgram.ProcessLoops(GetOriginalImplementation(originalProgram));
                Block header = FindLoopHeaderWithStateName(conflictingStatePrefix, blockGraph);
                Debug.Assert(header != null);
                HashSet<Block> loopNodes = new HashSet<Block>(
                  blockGraph.BackEdgeNodes(header).Select(item => blockGraph.NaturalLoops(header, item)).SelectMany(item => item));
                return GetSourceLocationsFromBlocks("_CHECK_" + accessType + "_" + arrayName, loopNodes);
            }
            else if (conflictingState.Contains("call_return_state"))
            {
                return GetSourceLocationsFromCall(
                    "_CHECK_" + accessType + "_" + arrayName,
                    QKeyValue.FindStringAttribute(conflictingAction.Attributes, "procedureName"));
            }
            else
            {
                Debug.Assert(conflictingState.Contains("check_state"));
                return new HashSet<SourceLocationInfo>
                    { new SourceLocationInfo(conflictingAction.Attributes, GetSourceFileName(), conflictingAction.tok) };
            }
        }

        private static Block FindLoopHeaderWithStateName(string stateName, Graph<Block> cfg)
        {
            foreach (var b in cfg.Headers)
            {
                foreach (var c in b.Cmds.OfType<AssumeCmd>())
                {
                    var stateId = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
                    if (stateId == stateName)
                        return b;
                }
            }

            return null;
        }

        private Implementation GetOriginalImplementation(Program prog)
        {
            return prog.Implementations.Where(item => item.Name.Equals(impl.Name)).First();
        }

        private static Program GetOriginalProgram()
        {
            return Utilities.GetFreshProgram(CommandLineOptions.Clo.Files, false, false);
        }

        private AssumeCmd DetermineConflictingAction(CallCounterexample callCex, string raceyState, string accessHasOccurred, string accessOffset)
        {
            AssumeCmd lastLogAssume = null;
            BigInteger? lastOffsetValue = null;

            foreach (var b in callCex.Trace)
            {
                bool finished = false;
                foreach (var c in b.Cmds.OfType<AssumeCmd>())
                {
                    string stateName = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
                    if (stateName == null)
                        continue;

                    Model.CapturedState state = GetStateFromModel(stateName, callCex.Model);
                    if (state == null || state.TryGet(accessHasOccurred) is Model.Uninterpreted)
                    {
                        // Either the state was not recorded, or the state has nothing to do with the reported error, so do not
                        // analyse it further.
                        continue;
                    }

                    Model.Boolean ahoValue = state.TryGet(accessHasOccurred) as Model.Boolean;
                    Model.Element aoValue = RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.ORIGINAL
                        ? state.TryGet(accessOffset)
                        : callCex.Model.TryGetFunc(accessOffset).GetConstant();

                    if (!ahoValue.Value)
                    {
                        lastLogAssume = null;
                        lastOffsetValue = null;
                    }
                    else if (lastLogAssume == null || !ParseOffset(aoValue).Equals(lastOffsetValue))
                    {
                        lastLogAssume = c;
                        lastOffsetValue = ParseOffset(aoValue);
                    }

                    if (stateName.Equals(raceyState))
                    {
                        finished = true;
                    }

                    break;
                }

                if (finished)
                {
                    break;
                }
            }

            Debug.Assert(lastLogAssume != null);
            return lastLogAssume;
        }

        private static IEnumerable<SourceLocationInfo> GetSourceLocationsFromCall(string checkProcedureName, string calleeName)
        {
            Program originalProgram = GetOriginalProgram();
            var bodies = originalProgram.Implementations.Where(item => item.Name.Equals(calleeName)).ToList();
            if (bodies.Count == 0)
            {
                return Enumerable.Empty<SourceLocationInfo>();
            }

            return GetSourceLocationsFromBlocks(checkProcedureName, bodies[0].Blocks);
        }

        private static IEnumerable<SourceLocationInfo> GetSourceLocationsFromBlocks(string checkProcedureName, IEnumerable<Block> blocks)
        {
            HashSet<SourceLocationInfo> possibleSources = new HashSet<SourceLocationInfo>();
            foreach (var c in blocks.Select(item => item.Cmds).SelectMany(item => item).OfType<CallCmd>())
            {
                if (c.callee.Equals(checkProcedureName))
                {
                    possibleSources.Add(new SourceLocationInfo(c.Attributes, GetSourceFileName(), c.tok));
                }
                else
                {
                    possibleSources.UnionWith(GetSourceLocationsFromCall(checkProcedureName, c.callee));
                }
            }

            return possibleSources;
        }

        private static Model.CapturedState GetStateFromModel(string stateName, Model m)
        {
            Model.CapturedState state = null;
            foreach (var s in m.States)
            {
                if (s.Name.Equals(stateName))
                {
                    state = s;
                    break;
                }
            }

            return state;
        }

        private static Variable ExtractAccessHasOccurredVar(CallCounterexample err)
        {
            var vfv = new VariableFinderVisitor(
                RaceInstrumentationUtil.MakeHasOccurredVariableName(QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array"), GetAccessType(err)));
            vfv.Visit(err.FailingRequires.Condition);
            return vfv.GetVariable();
        }

        private static Variable ExtractOffsetVar(CallCounterexample err)
        {
            var vfv = new VariableFinderVisitor(
                RaceInstrumentationUtil.MakeOffsetVariableName(QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array"), GetAccessType(err)));
            vfv.Visit(err.FailingRequires.Condition);
            return vfv.GetVariable();
        }

        protected static AccessType GetAccessType(CallCounterexample err)
        {
            if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_write")
                || QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_read")
                || QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_atomic"))
            {
                return AccessType.WRITE;
            }
            else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_write")
                || QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_atomic"))
            {
                return AccessType.READ;
            }
            else
            {
                Debug.Assert(QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_read")
                    || QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_write"));
                return AccessType.ATOMIC;
            }
        }

        private static void ReportThreadSpecificFailure(AssertCounterexample err, string messagePrefix)
        {
            AssertCmd failingAssert = err.FailingAssert;

            Console.Error.WriteLine();
            var sli = new SourceLocationInfo(failingAssert.Attributes, GetSourceFileName(), failingAssert.tok);

            int relevantThread = QKeyValue.FindIntAttribute(failingAssert.Attributes, "thread", -1);
            Debug.Assert(relevantThread == 1 || relevantThread == 2);

            ErrorWriteLine(sli.Top() + ":", messagePrefix + " for " + ThreadDetails(err.Model, relevantThread, true), ErrorMsgType.Error);
            sli.PrintStackTrace();
            Console.Error.WriteLine();
        }

        private static void ReportFailingAssert(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "this assertion might not hold");
        }

        private static void ReportInvariantMaintedFailure(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "loop invariant might not be maintained by the loop");
        }

        private static void ReportInvariantEntryFailure(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "loop invariant might not hold on entry");
        }

        private static void ReportFailingBarrierInvariant(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "this barrier invariant might not hold");
        }

        private static void ReportFailingBarrierInvariantAccessCheck(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "insufficient permission may be held for evaluation of this barrier invariant");
        }

        private static void ReportFailingConstantWriteCheck(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "possible attempt to modify constant memory");
        }

        private static void ReportFailingBadPointerAccess(AssertCounterexample err)
        {
            ReportThreadSpecificFailure(err, "possible null pointer access");
        }

        private void ReportFailingArrayBounds(AssertCounterexample err)
        {
            PopulateModelWithStatesIfNecessary(err);

            string state = GetStateName(err);
            string arrayName = QKeyValue.FindStringAttribute(err.FailingAssert.Attributes, "array_name");
            Model.Element arrayOffset = GetStateFromModel(state, err.Model).TryGet("_ARRAY_OFFSET_" + arrayName);
            Axiom arrayInfo = GetOriginalProgram().Axioms.Where(item => QKeyValue.FindStringAttribute(item.Attributes, "array_info") == arrayName).First();

            string arrayAccess = GetArrayAccess(
                ParseOffset(arrayOffset),
                QKeyValue.FindStringAttribute(arrayInfo.Attributes, "source_name"),
                Convert.ToUInt32(QKeyValue.FindIntAttribute(arrayInfo.Attributes, "elem_width", -1)),
                Convert.ToUInt32(QKeyValue.FindIntAttribute(arrayInfo.Attributes, "source_elem_width", -1)),
                QKeyValue.FindStringAttribute(arrayInfo.Attributes, "source_dimensions").Split(','));

            var sli = new SourceLocationInfo(err.FailingAssert.Attributes, GetSourceFileName(), err.FailingAssert.tok);
            ErrorWriteLine(
                sli.Top() + ":",
                "possible array out-of-bounds access on array " + arrayAccess + " by " + ThreadDetails(err.Model, 2, false) + ":",
                ErrorMsgType.Error);
            sli.PrintStackTrace();
            Console.Error.WriteLine();
        }

        private BigInteger ParseOffset(Model.Element modelOffset)
        {
            BigInteger offset;
            if (modelOffset is Model.Number)
            {
                Model.Number numericOffset = (Model.Number)modelOffset;
                offset = BigInteger.Parse(numericOffset.Numeral);
            }
            else
            {
                Model.DatatypeValue datatypeOffset = (Model.DatatypeValue)modelOffset;
                if (datatypeOffset.ConstructorName != "-" || datatypeOffset.Arguments.Length != 1)
                {
                    throw new NotSupportedException("Unexpected offset");
                }

                Model.Number numericOffset = (Model.Number)datatypeOffset.Arguments[0];
                offset = -BigInteger.Parse(numericOffset.Numeral);

                while (offset < -BigInteger.Pow(2, sizeTBits - 1))
                {
                    offset += BigInteger.Pow(2, sizeTBits);
                }
            }

            while (offset >= BigInteger.Pow(2, sizeTBits - 1))
            {
                offset -= BigInteger.Pow(2, sizeTBits);
            }

            return offset;
        }

        private static string GetArrayAccess(BigInteger offset, string name, uint elWidth, uint srcElWidth, string[] dims)
        {
            Debug.Assert(elWidth != uint.MaxValue && elWidth % 8 == 0);
            Debug.Assert(srcElWidth != uint.MaxValue && srcElWidth % 8 == 0);

            elWidth /= 8;
            srcElWidth /= 8;

            uint[] dimStrides = new uint[dims.Count()];
            dimStrides[dims.Count() - 1] = 1;
            for (int i = dims.Count() - 2; i >= 0; i--)
                dimStrides[i] = dimStrides[i + 1] * Convert.ToUInt32(dims[i + 1]);

            BigInteger offsetInBytes = offset * elWidth;
            BigInteger leftoverBytes = offsetInBytes % srcElWidth;

            string arrayAccess = name;
            BigInteger remainder = offsetInBytes / srcElWidth;
            foreach (uint stride in dimStrides)
            {
                if (stride == 0)
                    return "0-sized array " + name;
                arrayAccess += "[" + (remainder / stride) + "]";
                remainder %= stride;
            }

            if (elWidth != srcElWidth)
            {
                if (elWidth == 1)
                    arrayAccess += " (byte " + leftoverBytes + ")";
                else
                    arrayAccess += " (bytes " + leftoverBytes + ".." + (leftoverBytes + elWidth - 1) + ")";
            }

            return arrayAccess;
        }

        private static void ReportEnsuresFailure(Ensures ensures)
        {
            Console.Error.WriteLine();
            var sli = new SourceLocationInfo(ensures.Attributes, GetSourceFileName(), ensures.tok);
            ErrorWriteLine(sli.Top() + ":", "postcondition might not hold on all return paths", ErrorMsgType.Error);
            sli.PrintStackTrace();
        }

        private static void ReportBarrierDivergence(CallCmd call)
        {
            Console.Error.WriteLine();
            var sli = new SourceLocationInfo(call.Attributes, GetSourceFileName(), call.tok);
            ErrorWriteLine(sli.Top() + ":", "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
            sli.PrintStackTrace();
        }

        private static void ReportRequiresFailure(CallCmd call, Requires req)
        {
            Console.Error.WriteLine();
            var callSLI = new SourceLocationInfo(call.Attributes, GetSourceFileName(), call.tok);
            var requiresSLI = new SourceLocationInfo(req.Attributes, GetSourceFileName(), req.tok);

            ErrorWriteLine(callSLI.Top() + ":", "a precondition for this call might not hold", ErrorMsgType.Error);
            callSLI.PrintStackTrace();

            ErrorWriteLine(requiresSLI.Top() + ":", "this is the precondition that might not hold", ErrorMsgType.Note);
            requiresSLI.PrintStackTrace();
        }

        private static void GetThreadsAndGroupsFromModel(
            Model model, int thread, out string localId, out string group, out string globalId, bool withSpaces)
        {
            localId = GetLocalId(model, withSpaces, thread);
            group = GetGroupId(model, withSpaces, thread);
            globalId = GetGlobalId(model, withSpaces, thread);
        }

        private static int GetGroupIdOneDimension(Model model, string dimension, int thread)
        {
            string name = "group_id_" + dimension;
            if (!((GVCommandLineOptions)CommandLineOptions.Clo).OnlyIntraGroupRaceChecking)
            {
                name += "$" + thread;
            }

            return model.TryGetFunc(name).GetConstant().AsInt();
        }

        private static int GetLocalIdOneDimension(Model model, string dimension, int thread)
        {
            return model.TryGetFunc("local_id_" + dimension + "$" + thread).GetConstant().AsInt();
        }

        private static int GetGroupSizeOneDimension(Model model, string dimension)
        {
            return model.TryGetFunc("group_size_" + dimension).GetConstant().AsInt();
        }

        private static int GetGlobalIdOneDimension(Model model, string dimension, int thread)
        {
            return (GetGroupIdOneDimension(model, dimension, thread) * GetGroupSizeOneDimension(model, dimension))
                + GetLocalIdOneDimension(model, dimension, thread);
        }

        private static string GetGroupId(Model model, bool withSpaces, int thread)
        {
            switch (((GVCommandLineOptions)CommandLineOptions.Clo).GridHighestDim)
            {
                case 0:
                    return string.Format(
                        "{0}",
                        GetGroupIdOneDimension(model, "x", thread));
                case 1:
                    return string.Format(
                        withSpaces ? "({0}, {1})" : "({0},{1})",
                        GetGroupIdOneDimension(model, "x", thread),
                        GetGroupIdOneDimension(model, "y", thread));
                case 2:
                    return string.Format(
                        withSpaces ? "({0}, {1}, {2})" : "({0},{1},{2})",
                        GetGroupIdOneDimension(model, "x", thread),
                        GetGroupIdOneDimension(model, "y", thread),
                        GetGroupIdOneDimension(model, "z", thread));
                default:
                    Debug.Assert(false, "GetGroupId(): Reached default case in switch over GridHighestDim.");
                    return string.Empty;
            }
        }

        private static string GetLocalId(Model model, bool withSpaces, int thread)
        {
            switch (((GVCommandLineOptions)CommandLineOptions.Clo).BlockHighestDim)
            {
                case 0:
                    return string.Format(
                        "{0}",
                        GetLocalIdOneDimension(model, "x", thread));
                case 1:
                    return string.Format(
                        withSpaces ? "({0}, {1})" : "({0},{1})",
                        GetLocalIdOneDimension(model, "x", thread),
                        GetLocalIdOneDimension(model, "y", thread));
                case 2:
                    return string.Format(
                        withSpaces ? "({0}, {1}, {2})" : "({0},{1},{2})",
                        GetLocalIdOneDimension(model, "x", thread),
                        GetLocalIdOneDimension(model, "y", thread),
                        GetLocalIdOneDimension(model, "z", thread));
                default:
                    Debug.Assert(false, "GetLocalId(): Reached default case in switch over BlockHighestDim.");
                    return string.Empty;
            }
        }

        private static string GetGlobalId(Model model, bool withSpaces, int thread)
        {
            switch (((GVCommandLineOptions)CommandLineOptions.Clo).BlockHighestDim)
            {
                case 0:
                    return string.Format(
                        "{0}",
                        GetGlobalIdOneDimension(model, "x", thread));
                case 1:
                    return string.Format(
                        withSpaces ? "({0}, {1})" : "({0},{1})",
                        GetGlobalIdOneDimension(model, "x", thread),
                        GetGlobalIdOneDimension(model, "y", thread));
                case 2:
                    return string.Format(
                        withSpaces ? "({0}, {1}, {2})" : "({0},{1},{2})",
                        GetGlobalIdOneDimension(model, "x", thread),
                        GetGlobalIdOneDimension(model, "y", thread),
                        GetGlobalIdOneDimension(model, "z", thread));
                default:
                    Debug.Assert(false, "GetGlobalId(): Reached default case in switch over BlockHighestDim.");
                    return string.Empty;
            }
        }

        protected static string GetArrayName(Requires requires)
        {
            string arrName = QKeyValue.FindStringAttribute(requires.Attributes, "array");
            Debug.Assert(arrName != null);
            Debug.Assert(arrName.StartsWith("$$"));
            return arrName;
        }

        private static string GetArraySourceName(Requires requires)
        {
            string arrName = QKeyValue.FindStringAttribute(requires.Attributes, "source_name");
            Debug.Assert(arrName != null);
            return arrName;
        }

        private static string ThreadDetails(Model model, int thread, bool withSpaces)
        {
            string localId, group, globalId;
            GetThreadsAndGroupsFromModel(model, thread, out localId, out group, out globalId, withSpaces);

            if (((GVCommandLineOptions)CommandLineOptions.Clo).SourceLanguage == SourceLanguage.CUDA)
            {
                return string.Format("thread {0} in thread block {1} (global id {2})", localId, group, globalId);
            }
            else
            {
                return string.Format("work item {0} with local id {1} in work group {2}", globalId, localId, group);
            }
        }

        private class VariableFinderVisitor : StandardVisitor
        {
            private string varName;
            private Variable variable = null;

            public VariableFinderVisitor(string varName)
            {
                this.varName = varName;
            }

            public override Variable VisitVariable(Variable node)
            {
                if (node.Name.Equals(varName))
                    variable = node;

                return base.VisitVariable(node);
            }

            internal Variable GetVariable()
            {
                return variable;
            }
        }
    }
}
