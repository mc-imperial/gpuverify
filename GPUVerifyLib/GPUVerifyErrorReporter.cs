//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;
using Microsoft.Basetypes;
using System.Text.RegularExpressions;
using System.Diagnostics.Contracts;


namespace GPUVerify {

  public class GPUVerifyErrorReporter {

    enum ErrorMsgType {
      Error,
      Note,
      NoError
    };

    private static void ErrorWriteLine(string locInfo, string message, ErrorMsgType msgtype) {
      Contract.Requires(message != null);
      ConsoleColor col = Console.ForegroundColor;
      if (!String.IsNullOrEmpty(locInfo)) {
        Console.Error.Write(locInfo + " ");
      }

      switch (msgtype) {
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

    private Implementation impl;

    internal GPUVerifyErrorReporter(Program program, string implName) {
      this.impl = program.Implementations().Where(Item => Item.Name.Equals(implName)).ToList()[0];
    }

    internal void ReportCounterexample(Counterexample error) {

      int WindowWidth;
      try {
        WindowWidth = Console.WindowWidth;
      } catch(IOException) {
        WindowWidth = 20;
      }

      for(int i = 0; i < WindowWidth; i++) {
        Console.Error.Write("-");
      }

      if (error is CallCounterexample) {
        CallCounterexample CallCex = (CallCounterexample)error;
        if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "barrier_divergence")) {
          ReportBarrierDivergence(CallCex.FailingCall);
        }
        else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "race")) {
          ReportRace(CallCex);
        }
        else {
          ReportRequiresFailure(CallCex.FailingCall, CallCex.FailingRequires);
        }
      }
      else if (error is ReturnCounterexample) {
        ReturnCounterexample ReturnCex = (ReturnCounterexample)error;
        ReportEnsuresFailure(ReturnCex.FailingEnsures);
      }
      else {
        AssertCounterexample AssertCex = (AssertCounterexample)error;
        if (AssertCex.FailingAssert is LoopInitAssertCmd) {
          ReportInvariantEntryFailure(AssertCex);
        }
        else if (AssertCex.FailingAssert is LoopInvMaintainedAssertCmd) {
          ReportInvariantMaintedFailure(AssertCex);
        }
        else if (QKeyValue.FindBoolAttribute(AssertCex.FailingAssert.Attributes, "barrier_invariant")) {
          ReportFailingBarrierInvariant(AssertCex);
        }
        else if (QKeyValue.FindBoolAttribute(AssertCex.FailingAssert.Attributes, "barrier_invariant_access_check")) {
          ReportFailingBarrierInvariantAccessCheck(AssertCex);
        }
        else if (QKeyValue.FindBoolAttribute(AssertCex.FailingAssert.Attributes, "constant_write")) {
          ReportFailingConstantWriteCheck(AssertCex);
        }
        else if (QKeyValue.FindBoolAttribute(AssertCex.FailingAssert.Attributes, "bad_pointer_access")) {
          ReportFailingBadPointerAccess(AssertCex);
        }
        else {
          ReportFailingAssert(AssertCex);
        }
      }

      DisplayParameterValues(error);
    }

    private void DisplayParameterValues(Counterexample error)
    {
      if (impl.InParams.Count() == 0)
      {
        return;
      }

      Console.Error.WriteLine("Bitwise values of parameters of '" + DemangleName(impl.Name.TrimStart(new char[] { '$' })) + "':");
      PopulateModelWithStatesIfNecessary(error);

      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(error.Model, out thread1, out thread2, out group1, out group2, false);
      foreach (var p in impl.InParams)
      {
        int id;
        string stripped = GVUtil.StripThreadIdentifier(p.Name, out id).TrimStart(new char[] { '$' });
        Console.Error.Write("  " + stripped + " = ");

        var func = error.Model.TryGetFunc(p.Name);
        if (func != null)
        {
          var val = func.GetConstant();
          if (val is Model.BitVector)
          {
            Console.Error.Write(((Model.BitVector)val).Numeral);
          }
          else if (val is Model.Uninterpreted)
          {
            Console.Error.Write("<irrelevant>");
          }
          else
          {
            Console.Error.Write("<unknown>");
          }
        }
        else
        {
          Console.Error.Write("<unknown>");
        }
        Console.Error.WriteLine(id == 1 ? " (" + SpecificNameForThread() + " " + thread1 + ", " + SpecificNameForGroup() + " " + group1 + ")" :
                               (id == 2 ? " (" + SpecificNameForThread() + " " + thread2 + ", " + SpecificNameForGroup() + " " + group2 + ")" : ""));
      }
      Console.Error.WriteLine();
    }

    private void ReportRace(CallCounterexample CallCex) {

      string raceName, access1, access2;

      DetermineNatureOfRace(CallCex, out raceName, out access1, out access2);

      PopulateModelWithStatesIfNecessary(CallCex);

      string RaceyArrayName = GetArrayName(CallCex.FailingRequires);
      Debug.Assert(RaceyArrayName != null);

      IEnumerable<SourceLocationInfo> PossibleSourcesForFirstAccess = GetPossibleSourceLocationsForFirstAccessInRace(CallCex, RaceyArrayName, AccessType.Create(access1),
        GetStateName(CallCex));
      SourceLocationInfo SourceInfoForSecondAccess = new SourceLocationInfo(GetAttributes(CallCex.FailingCall), GetSourceFileName(), CallCex.FailingCall.tok);

      ulong RaceyOffset = GetOffsetInBytes(CallCex);

      ErrorWriteLine("\n" + SourceInfoForSecondAccess.Top().GetFile() + ":", "possible " + raceName + " race on ((char*)" + 
        CleanUpArrayName(DemangleName(RaceyArrayName)) + ")[" + RaceyOffset + "]:\n", ErrorMsgType.Error);

      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(CallCex.Model, out thread1, out thread2, out group1, out group2, true);

      Console.Error.WriteLine(access2 + " by " + SpecificNameForThread() + " " + thread2 + " in " + SpecificNameForGroup() + " " + group2 + ", " + SourceInfoForSecondAccess.Top() + ":");
      SourceInfoForSecondAccess.PrintStackTrace();

      Console.Error.Write(access1 + " by " + SpecificNameForThread() + " " + thread1 + " in " + SpecificNameForGroup() + " " + group1 + ", ");
      if(PossibleSourcesForFirstAccess.Count() == 1) {
        Console.Error.WriteLine(PossibleSourcesForFirstAccess.ToList()[0].Top() + ":");
        PossibleSourcesForFirstAccess.ToList()[0].PrintStackTrace();
      } else if(PossibleSourcesForFirstAccess.Count() == 0) {
        Console.Error.WriteLine("from external source location\n");
      } else {
        Console.Error.WriteLine("possible sources are:");
        List<SourceLocationInfo> LocationsAsList = PossibleSourcesForFirstAccess.ToList();
        LocationsAsList.Sort(new SourceLocationInfo.SourceLocationInfoComparison());
        foreach(var sli in LocationsAsList) {
          Console.Error.WriteLine(sli.Top() + ":");
          sli.PrintStackTrace();
        }
        Console.Error.WriteLine();
      }
    }

    private static string GetStateName(CallCounterexample CallCex)
    {
      Contract.Requires(QKeyValue.FindStringAttribute(CallCex.FailingCall.Attributes, "check_id") != null);
      string CheckId = QKeyValue.FindStringAttribute(CallCex.FailingCall.Attributes, "check_id");
      return QKeyValue.FindStringAttribute(
        (CallCex.Trace.Last().Cmds.OfType<AssumeCmd>().Where(
          Item => QKeyValue.FindStringAttribute(Item.Attributes, "check_id") == CheckId).ToList()[0]
        ).Attributes, "captureState");
    }

    private static string GetSourceFileName()
    {
      return CommandLineOptions.Clo.Files[CommandLineOptions.Clo.Files.Count() - 1];
    }

    private string CleanUpArrayName(string name)
    {
      // The purpose of this method is to take a demangled array name
      // and turn it into a more readable form.
      // The issue motivating this is that in CUDA, __shared__ arrays
      // declared in functions get mangled to include the enclosing
      // function.  The demangled name contains the whole of the
      // function signature, which is not easy to read.

      if(((GVCommandLineOptions)CommandLineOptions.Clo).SourceLanguage == SourceLanguage.OpenCL) {
        return name;
      }

      // Check to see whether the name includes a function open parenthesis
      if(!name.Contains(":")) {
        return name;
      }
      
      return String.Join("::",
        Regex.Replace(Regex.Replace(name.Split(new char[] { ' ' }).Last(), "'", ""), "`", "").
          Split(new string[] { "::" }, StringSplitOptions.RemoveEmptyEntries).
            Where(Item => !Regex.IsMatch(Item, @"^\d+$")).ToArray());
    }

    private static void PopulateModelWithStatesIfNecessary(Counterexample Cex)
    {
      if (!Cex.ModelHasStatesAlready)
      {
        Cex.PopulateModelWithStates();
        Cex.ModelHasStatesAlready = true;
      }
    }

    private static void DetermineNatureOfRace(CallCounterexample CallCex, out string raceName, out string access1, out string access2)
    {
      if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_read"))
      {
        raceName = "write-read";
        access1 = "Write";
        access2 = "Read";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_write"))
      {
        raceName = "read-write";
        access1 = "Read";
        access2 = "Write";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_write"))
      {
        raceName = "write-write";
        access1 = "Write";
        access2 = "Write";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_read"))
      {
        raceName = "atomic-read";
        access1 = "Atomic";
        access2 = "Read";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_write"))
      {
        raceName = "atomic-write";
        access1 = "Atomic";
        access2 = "Write";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_atomic"))
      {
        raceName = "read-atomic";
        access1 = "Read";
        access2 = "Atomic";
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_atomic"))
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

    private IEnumerable<SourceLocationInfo> GetPossibleSourceLocationsForFirstAccessInRace(CallCounterexample CallCex, string ArrayName, AccessType AccessType, string RaceyState)
    {
      string AccessHasOccurred = RaceInstrumentationUtil.MakeHasOccurredVariableName("$$" + ArrayName, AccessType);
      string AccessOffset = RaceInstrumentationUtil.MakeOffsetVariableName("$$" + ArrayName, AccessType);

      AssumeCmd ConflictingAction = DetermineConflictingAction(CallCex, RaceyState, AccessHasOccurred, AccessOffset);

      var ConflictingState = QKeyValue.FindStringAttribute(ConflictingAction.Attributes, "captureState");

      if (ConflictingState.Contains("loop_head_state"))
      {
        Program originalProgram = GVUtil.GetFreshProgram(CommandLineOptions.Clo.Files, true, false);
        Implementation originalImplementation = originalProgram.Implementations().Where(Item => Item.Name.Equals(impl.Name)).ToList()[0];
        var blockGraph = originalProgram.ProcessLoops(originalImplementation);
        Block header = null;
        foreach (var b in blockGraph.Headers)
        {
          foreach (var c in b.Cmds.OfType<AssumeCmd>())
          {
            var stateId = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
            if (stateId != null && stateId.Equals(ConflictingState))
            {
              header = b;
              break;
            }
          }
          if (header != null)
          {
            break;
          }
        }
        Debug.Assert(header != null);
        HashSet<Block> LoopNodes = new HashSet<Block>(
          blockGraph.BackEdgeNodes(header).Select(Item => blockGraph.NaturalLoops(header, Item)).SelectMany(Item => Item)
        );
        return GetSourceLocationsFromBlocks("_CHECK_" + AccessType + "_$$" + ArrayName, LoopNodes);
      }
      else if(ConflictingState.Contains("call_return_state")  ) {
        return GetSourceLocationsFromCall("_CHECK_" + AccessType + "_$$" + ArrayName, 
          QKeyValue.FindStringAttribute(ConflictingAction.Attributes, "procedureName"));
      } else {
        Debug.Assert(ConflictingState.Contains("check_state"));
        return new HashSet<SourceLocationInfo> { 
          new SourceLocationInfo(ConflictingAction.Attributes, GetSourceFileName(), ConflictingAction.tok)
        };
      }
    }

    private static AssumeCmd DetermineConflictingAction(CallCounterexample CallCex, string RaceyState, string AccessHasOccurred, string AccessOffset)
    {
      AssumeCmd LastLogAssume = null;
      string LastOffsetValue = null;

      foreach (var b in CallCex.Trace)
      {
        bool finished = false;
        foreach (var c in b.Cmds.OfType<AssumeCmd>())
        {
          string StateName = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
          if (StateName == null)
          {
            continue;
          }
          Model.CapturedState state = GetStateFromModel(StateName, CallCex.Model);
          if (state == null || state.TryGet(AccessHasOccurred) is Model.Uninterpreted)
          {
            // Either the state was not recorded, or the state has nothing to do with the reported error, so do not
            // analyse it further.
            continue;
          }

          Model.Boolean AHO_value = state.TryGet(AccessHasOccurred) as Model.Boolean;
          Model.BitVector AO_value = 
            (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.STANDARD
            ? state.TryGet(AccessOffset)
            : CallCex.Model.TryGetFunc(AccessOffset).GetConstant()) as Model.BitVector;

          if (!AHO_value.Value)
          {
            LastLogAssume = null;
            LastOffsetValue = null;
          }
          else if (LastLogAssume == null || !AO_value.Numeral.Equals(LastOffsetValue))
          {
            LastLogAssume = c;
            LastOffsetValue = AO_value.Numeral;
          }
          if (StateName.Equals(RaceyState))
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

      Debug.Assert(LastLogAssume != null);
      return LastLogAssume;
    }

    private static IEnumerable<SourceLocationInfo> GetSourceLocationsFromCall(string CheckProcedureName, string CalleeName)
    {
      Program originalProgram = GVUtil.GetFreshProgram(CommandLineOptions.Clo.Files, true, false);
      var Bodies =  originalProgram.Implementations().Where(Item => Item.Name.Equals(CalleeName)).ToList();
      if(Bodies.Count == 0) {
        return new HashSet<SourceLocationInfo>();
      }
      return GetSourceLocationsFromBlocks(CheckProcedureName, Bodies[0].Blocks);
    }

    private static IEnumerable<SourceLocationInfo> GetSourceLocationsFromBlocks(string CheckProcedureName, IEnumerable<Block> Blocks)
    {
      HashSet<SourceLocationInfo> PossibleSources = new HashSet<SourceLocationInfo>();
      foreach (var c in Blocks.Select(Item => Item.Cmds).SelectMany(Item => Item).OfType<CallCmd>())
      {
        if (c.callee.Equals(CheckProcedureName))
        {
          PossibleSources.Add(new SourceLocationInfo(c.Attributes, GetSourceFileName(), c.tok));
        } else {
          foreach(var sl in GetSourceLocationsFromCall(CheckProcedureName, c.callee)) {
            PossibleSources.Add(sl);
          }
        }
      }
      return PossibleSources;
    }

    private static Model.CapturedState GetStateFromModel(string StateName, Model m)
    {
      Model.CapturedState state = null;
      foreach (var s in m.States)
      {
        if (s.Name.Equals(StateName))
        {
          state = s;
          break;
        }
      }
      return state;
    }

    private static ulong GetOffsetInBytes(CallCounterexample Cex) {
      uint ElemWidth = (uint)QKeyValue.FindIntAttribute(ExtractAccessHasOccurredVar(Cex).Attributes, "elem_width", int.MaxValue);
      Debug.Assert(ElemWidth != int.MaxValue);
      var element =
        (RaceInstrumentationUtil.RaceCheckingMethod == RaceCheckingMethod.STANDARD
        ? GetStateFromModel(GetStateName(Cex),
           Cex.Model).TryGet(ExtractOffsetVar(Cex).Name)
        : Cex.Model.TryGetFunc(ExtractOffsetVar(Cex).Name).GetConstant()) as Model.Number;
      return (Convert.ToUInt64(element.Numeral) * ElemWidth) / 8;
    }

    private static Variable ExtractAccessHasOccurredVar(CallCounterexample err) {
      var VFV = new VariableFinderVisitor(
        RaceInstrumentationUtil.MakeHasOccurredVariableName(QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array"), GetAccessType(err)));
      VFV.Visit(err.FailingRequires.Condition);
      return VFV.GetVariable();
    }

    private static Variable ExtractOffsetVar(CallCounterexample err) {
      var VFV = new VariableFinderVisitor(
        RaceInstrumentationUtil.MakeOffsetVariableName(QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array"), GetAccessType(err)));
      VFV.Visit(err.FailingRequires.Condition);
      return VFV.GetVariable();
    }

    private static AccessType GetAccessType(CallCounterexample err)
    {
      if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_write") ||
          QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_read") ||
          QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_atomic"))
      {
        return AccessType.WRITE;
      }
      else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_write") ||
               QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_atomic"))
      {
        return AccessType.READ;
      }
      else
      {
        Debug.Assert(QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_read") ||
                     QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_write"));
        return AccessType.ATOMIC;
      }
    }

    static QKeyValue GetAttributes(Absy a) {
      if (a is PredicateCmd) {
        return (a as PredicateCmd).Attributes;
      }
      else if (a is Requires) {
        return (a as Requires).Attributes;
      }
      else if (a is Ensures) {
        return (a as Ensures).Attributes;
      }
      else if (a is CallCmd) {
        return (a as CallCmd).Attributes;
      }
      //Debug.Assert(false);
      return null;
    }

    private static void ReportThreadSpecificFailure(AssertCounterexample err, string messagePrefix) {
      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(err.Model, out thread1, out thread2, out group1, out group2, true);

      AssertCmd failingAssert = err.FailingAssert;

      Console.Error.WriteLine();
      var sli = new SourceLocationInfo(GetAttributes(failingAssert), GetSourceFileName(), failingAssert.tok);

      int relevantThread = QKeyValue.FindIntAttribute(GetAttributes(failingAssert), "thread", -1);
      Debug.Assert(relevantThread == 1 || relevantThread == 2);

      ErrorWriteLine(sli.Top() + ":", messagePrefix + " for " + SpecificNameForThread() + " " +
                     (relevantThread == 1 ? thread1 : thread2) + " in " + SpecificNameForGroup() + " " + (relevantThread == 1 ? group1 : group2), ErrorMsgType.Error);
      sli.PrintStackTrace();
      Console.Error.WriteLine();
    }

    private static void ReportFailingAssert(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "this assertion might not hold");
    }

    private static void ReportInvariantMaintedFailure(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "loop invariant might not be maintained by the loop");
    }

    private static void ReportInvariantEntryFailure(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "loop invariant might not hold on entry");
    }

    private static void ReportFailingBarrierInvariant(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "this barrier invariant might not hold");
    }

    private static void ReportFailingBarrierInvariantAccessCheck(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "insufficient permission may be held for evaluation of this barrier invariant");
    }

    private static void ReportFailingConstantWriteCheck(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "possible attempt to modify constant memory");
    }

    private static void ReportFailingBadPointerAccess(AssertCounterexample err) {
      ReportThreadSpecificFailure(err, "possible null pointer access");
    }

    private static void ReportEnsuresFailure(Absy node) {
      Console.Error.WriteLine();
      var sli = new SourceLocationInfo(GetAttributes(node), GetSourceFileName(), node.tok);
      ErrorWriteLine(sli.Top() + ":", "postcondition might not hold on all return paths", ErrorMsgType.Error);
      sli.PrintStackTrace();
    }

    private static void ReportBarrierDivergence(Absy node) {
      Console.Error.WriteLine();
      var sli = new SourceLocationInfo(GetAttributes(node), GetSourceFileName(), node.tok);
      ErrorWriteLine(sli.Top() + ":", "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      sli.PrintStackTrace();
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode) {
      Console.Error.WriteLine();
      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), GetSourceFileName(), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), GetSourceFileName(), reqNode.tok);

      ErrorWriteLine(CallSLI.Top() + ":", "a precondition for this call might not hold", ErrorMsgType.Error);
      CallSLI.PrintStackTrace();

      ErrorWriteLine(RequiresSLI.Top() + ":", "this is the precondition that might not hold", ErrorMsgType.Note);
      RequiresSLI.PrintStackTrace();
    }

    private static void GetThreadsAndGroupsFromModel(Model model, out string thread1, out string thread2, out string group1, out string group2, bool withSpaces) {
      thread1 = GetThreadOne(model, withSpaces);
      thread2 = GetThreadTwo(model, withSpaces);
      group1 = GetGroup(model, withSpaces, 1);
      group2 = GetGroup(model, withSpaces, 2);
    }

    private static string GetGroup(Model model, bool withSpaces, int thread) {
      switch (((GVCommandLineOptions)CommandLineOptions.Clo).GridHighestDim) {
        case 0:
        return ""
          + GetGid(model, "x", thread);
        case 1:
        return "("
          + GetGid(model, "x", thread)
            + "," + (withSpaces ? " " : "")
            + GetGid(model, "y", thread)
            + ")";
        case 2:
        return "("
          + GetGid(model, "x", thread)
            + "," + (withSpaces ? " " : "")
            + GetGid(model, "y", thread)
            + "," + (withSpaces ? " " : "")
            + GetGid(model, "z", thread)
            + ")";
        default:
        Debug.Assert(false, "GetGroup(): Reached default case in switch over GridHighestDim.");
        return "";
      }
    }

    private static int GetGid(Model model, string dimension, int thread) {
      string name = "group_id_" + dimension;
      if(!((GVCommandLineOptions)CommandLineOptions.Clo).OnlyIntraGroupRaceChecking) {
        name += "$" + thread;
      }

      return model.TryGetFunc(name).GetConstant().AsInt();
    }

    private static string GetThreadTwo(Model model, bool withSpaces) {
      switch (((GVCommandLineOptions)CommandLineOptions.Clo).BlockHighestDim) {
        case 0:
        return ""
          + GetLidX2(model);
        case 1:
        return "("
          + GetLidX2(model)
            + "," + (withSpaces ? " " : "")
            + GetLidY2(model)
            + ")";
        case 2:
        return "("
          + GetLidX2(model)
            + "," + (withSpaces ? " " : "")
            + GetLidY2(model)
            + "," + (withSpaces ? " " : "")
            + GetLidZ2(model)
            + ")";
        default:
        Debug.Assert(false, "GetThreadTwo(): Reached default case in switch over BlockHighestDim.");
        return "";
      }
    }


    private static int GetLidZ2(Model model) {
      return model.TryGetFunc("local_id_z$2").GetConstant().AsInt();
    }

    private static int GetLidY2(Model model) {
      return model.TryGetFunc("local_id_y$2").GetConstant().AsInt();
    }

    private static int GetLidX2(Model model) {
      return model.TryGetFunc("local_id_x$2").GetConstant().AsInt();
    }

    private static string GetThreadOne(Model model, bool withSpaces) {
      switch (((GVCommandLineOptions)CommandLineOptions.Clo).BlockHighestDim) {
        case 0:
        return "" 
          + model.TryGetFunc("local_id_x$1").GetConstant().AsInt();
        case 1:
        return "("
          + model.TryGetFunc("local_id_x$1").GetConstant().AsInt()
            + "," + (withSpaces ? " " : "")
            + model.TryGetFunc("local_id_y$1").GetConstant().AsInt()
            + ")";
      case 2:
        return "("
          + model.TryGetFunc("local_id_x$1").GetConstant().AsInt()
            + "," + (withSpaces ? " " : "")
            + model.TryGetFunc("local_id_y$1").GetConstant().AsInt()
            + "," + (withSpaces ? " " : "")
            + model.TryGetFunc("local_id_z$1").GetConstant().AsInt()
            + ")";
        default:
        Debug.Assert(false, "GetThreadOne(): Reached default case in switch over BlockHighestDim.");
        return "";
      }
    }

    private static string GetArrayName(Requires requires) {
      string arrName = QKeyValue.FindStringAttribute(requires.Attributes, "array");
      Debug.Assert(arrName != null);
      Debug.Assert(arrName.StartsWith("$$"));
      return arrName.Substring("$$".Length);
    }

    public static void FixStateIds(Program Program) {
      new StateIdFixer().FixStateIds(Program);
    }

    private static string SpecificNameForGroup() {
      if(((GVCommandLineOptions)CommandLineOptions.Clo).SourceLanguage == SourceLanguage.CUDA) {
        return "block";
      } else {
        return "work group";
      }
    }

    private static string SpecificNameForThread() {
      if(((GVCommandLineOptions)CommandLineOptions.Clo).SourceLanguage == SourceLanguage.CUDA) {
        return "thread";
      } else {
        return "work item";
      }
    }

    private static string DemangleName(string name) {
      var gvClo = CommandLineOptions.Clo as GVCommandLineOptions;
      if(gvClo.SourceLanguage == SourceLanguage.CUDA && gvClo.DemanglerPath != null) {
        try {
          name = name.Replace('~','@');
          Process demangler = new Process();
          demangler.StartInfo = new ProcessStartInfo(gvClo.DemanglerPath, "-l cu " + name);
          demangler.StartInfo.UseShellExecute = false;
          demangler.StartInfo.RedirectStandardOutput = true;
          string demangled = "";
          demangler.OutputDataReceived += (sender, args) => demangled += args.Data;
          demangler.Start();
          demangler.BeginOutputReadLine();
          demangler.WaitForExit();
          return demangled;
        } catch(Exception e) {
          Console.Error.WriteLine("warning: name demangling failed with: " + e.Message);
          return name;
        }
      }
      return name;
    }

  }

  class StateIdFixer {

    private int FreshCounter = 0;

    internal void FixStateIds(Program Program) {
      foreach(var impl in Program.Implementations()) {
        impl.Blocks = new List<Block>(impl.Blocks.Select(FixStateIds));
      }
    }

    private Block FixStateIds(Block b) {
      List<Cmd> newCmds = new List<Cmd>();
      for (int i = 0; i < b.Cmds.Count(); i++) {
        var a = b.Cmds[i] as AssumeCmd;
        if (a != null && (QKeyValue.FindStringAttribute(a.Attributes, "captureState") != null)) {
          string StateName = QKeyValue.FindStringAttribute(a.Attributes, "captureState");
          newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, FreshStateId(a.Attributes)));
        }
        else {
          newCmds.Add(b.Cmds[i]);
        }
      }
      b.Cmds = newCmds;
      return b;
    }

    private QKeyValue FreshStateId(QKeyValue Attributes) {
      // Returns attributes identical to Attributes, but:
      // - reversed (for ease of implementation; should not matter)
      // - with the value for "captureState" replaced by a fresh value
      Contract.Requires(QKeyValue.FindStringAttribute(Attributes, "captureState") != null);
      string FreshValue = QKeyValue.FindStringAttribute(Attributes, "captureState") + "$renamed$" + FreshCounter;
      FreshCounter++;

      QKeyValue result = null;
      while (Attributes != null) {
        if (Attributes.Key.Equals("captureState")) {
          result = new QKeyValue(Token.NoToken, Attributes.Key, new List<object>() { FreshValue }, result);
        } else {
          result = new QKeyValue(Token.NoToken, Attributes.Key, Attributes.Params, result);
        }
        Attributes = Attributes.Next;
      }
      return result;
    }

  }

}
