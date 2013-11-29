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

      Console.Error.WriteLine("Bitwise values of parameters of " + impl.Name.TrimStart(new char[] { '$' }) + ":");
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
        Console.Error.WriteLine(id == 1 ? " (thread " + thread1 + ", group " + group1 + ")" :
                               (id == 2 ? " (thread " + thread2 + ", group " + group2 + ")" : ""));
      }
      Console.WriteLine();
    }

    private void ReportRace(CallCounterexample CallCex) {

      string raceName, access1, access2;

      DetermineNatureOfRace(CallCex, out raceName, out access1, out access2);

      PopulateModelWithStatesIfNecessary(CallCex);

      string RaceyArrayName = GetArrayName(CallCex.FailingRequires);
      Debug.Assert(RaceyArrayName != null);

      IEnumerable<SourceLocationInfo> PossibleSourcesForFirstAccess = GetPossibleSourceLocationsForFirstAccessInRace(CallCex, RaceyArrayName, access1,
        QKeyValue.FindStringAttribute(CallCex.FailingCall.Attributes, "state_id"));
      SourceLocationInfo SourceInfoForSecondAccess = new SourceLocationInfo(GetAttributes(CallCex.FailingCall), CallCex.FailingCall.tok);

      uint RaceyOffset = GetOffsetInBytes(ExtractOffsetVar(CallCex), CallCex.Model, CallCex.FailingCall);

      ErrorWriteLine("\n" + SourceInfoForSecondAccess.GetFile() + ":", "possible " + raceName + " race on ((char*)" + 
        DemangleName(RaceyArrayName) + ")[" + RaceyOffset + "]:\n", ErrorMsgType.Error);

      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(CallCex.Model, out thread1, out thread2, out group1, out group2, true);

      Console.Error.WriteLine(access2 + " by thread " + thread2 + " in group " + group2 + ", " + SourceInfoForSecondAccess);
      GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(SourceInfoForSecondAccess.FetchCodeLine() + "\n", 2));

      Console.Error.Write(access1 + " by thread " + thread1 + " in group " + group1 + ", ");
      if(PossibleSourcesForFirstAccess.Count() == 1) {
        Console.Error.WriteLine(PossibleSourcesForFirstAccess.ToList()[0]);
        GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(PossibleSourcesForFirstAccess.ToList()[0].FetchCodeLine() + "\n", 2));
      } else if(PossibleSourcesForFirstAccess.Count() == 0) {
        Console.Error.WriteLine("from external source location\n");
      } else {
        Console.Error.WriteLine("possible sources are:");
        List<SourceLocationInfo> LocationsAsList = PossibleSourcesForFirstAccess.ToList();
        LocationsAsList.Sort(new SourceLocationInfo.SourceLocationInfoComparison());
        foreach(var sli in LocationsAsList) {
          Console.Error.WriteLine(sli);
          GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(sli.FetchCodeLine(), 2));
        }
        Console.WriteLine();
      }
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

    private IEnumerable<SourceLocationInfo> GetPossibleSourceLocationsForFirstAccessInRace(CallCounterexample CallCex, string ArrayName, string AccessType, string RaceyState)
    {
      string ACCESS_HAS_OCCURRED = "_" + AccessType.ToUpper() + "_HAS_OCCURRED_$$" + ArrayName;
      string ACCESS_OFFSET = "_" + AccessType.ToUpper() + "_OFFSET_$$" + ArrayName;

      AssumeCmd ConflictingAction = DetermineConflictingAction(CallCex, RaceyState, ACCESS_HAS_OCCURRED, ACCESS_OFFSET);

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
        return GetSourceLocationsFromBlocks("_CHECK_" + AccessType.ToUpper() + "_$$" + ArrayName, LoopNodes);
      }
      else if(ConflictingState.Contains("call_return_state")  ) {
        return GetSourceLocationsFromCall("_CHECK_" + AccessType.ToUpper() + "_$$" + ArrayName, 
          QKeyValue.FindStringAttribute(ConflictingAction.Attributes, "procedureName"));
      } else {
        Debug.Assert(ConflictingState.Contains("check_state"));
        return new HashSet<SourceLocationInfo> { 
          new SourceLocationInfo(ConflictingAction.Attributes, ConflictingAction.tok)
        };
      }
    }

    private static AssumeCmd DetermineConflictingAction(CallCounterexample CallCex, string RaceyState, string ACCESS_HAS_OCCURRED, string ACCESS_OFFSET)
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
          if (state == null || state.TryGet(ACCESS_HAS_OCCURRED) is Model.Uninterpreted)
          {
            // Either the state was not recorded, or the state has nothing to do with the reported error, so do not
            // analyse it further.
            continue;
          }

          Model.Boolean AHO_value = state.TryGet(ACCESS_HAS_OCCURRED) as Model.Boolean;
          Model.BitVector AO_value = state.TryGet(ACCESS_OFFSET) as Model.BitVector;
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
          PossibleSources.Add(new SourceLocationInfo(c.Attributes, c.tok));
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

    private static uint GetOffsetInBytes(Variable OffsetVar, Model Model, CallCmd FailingCall) {
      var element = GetStateFromModel(QKeyValue.FindStringAttribute(FailingCall.Attributes, "state_id"), Model).TryGet(OffsetVar.Name) as Model.Number;
      uint elemOffset = Convert.ToUInt32(element.Numeral);
      Debug.Assert(OffsetVar.Attributes != null);
      uint elemWidth = (uint)QKeyValue.FindIntAttribute(OffsetVar.Attributes, "elem_width", int.MaxValue);
      Debug.Assert(elemWidth != int.MaxValue);
      return (elemOffset * elemWidth) / 8;
    }

    private static Variable ExtractOffsetVar(CallCounterexample err) {
      // The offset variable name can be exactly reconstructed from the attributes of the requires clause
      string ArrayName = QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array");
      AccessType Access;
      if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_write") ||
          QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_read") ||
          QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_atomic")) {
        Access = AccessType.WRITE;
      }
      else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_write") ||
               QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_atomic")) {
        Access = AccessType.READ;
      }
      else {
        Debug.Assert(QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_read") ||
                     QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "atomic_write"));
        Access = AccessType.ATOMIC;
      }

      string OffsetVarName = "_" + Access + "_OFFSET_" + ArrayName;

      var VFV = new VariableFinderVisitor(OffsetVarName);
      VFV.Visit(err.FailingRequires.Condition);
      return VFV.GetVariable();
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

      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(failingAssert), failingAssert.tok);

      int relevantThread = QKeyValue.FindIntAttribute(GetAttributes(failingAssert), "thread", -1);
      Debug.Assert(relevantThread == 1 || relevantThread == 2);

      ErrorWriteLine(sli.ToString(), messagePrefix + " for thread " +
                     (relevantThread == 1 ? thread1 : thread2) + " in group " + (relevantThread == 1 ? group1 : group2), ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(sli.FetchCodeLine());
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
      Console.WriteLine();
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "postcondition might not hold on all return paths", ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportBarrierDivergence(Absy node) {
      Console.WriteLine();
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode) {
      Console.WriteLine();
      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      ErrorWriteLine(CallSLI.ToString(), "a precondition for this call might not hold", ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine(), 2));

      ErrorWriteLine(RequiresSLI.ToString(), "this is the precondition that might not hold", ErrorMsgType.Note);
      GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine(), 2));
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

    private static string TrimLeadingSpaces(string s1, int noOfSpaces) {
      if (String.IsNullOrWhiteSpace(s1)) {
        return s1;
      }

      int index;
      for (index = 0; (index + 1) < s1.Length && Char.IsWhiteSpace(s1[index]); ++index) ;
      string returnString = s1.Substring(index);
      for (int i = noOfSpaces; i > 0; --i) {
        returnString = " " + returnString;
      }
      return returnString;
    }

    public static void FixStateIds(Program Program) {
      new StateIdFixer().FixStateIds(Program);
    }

    private static void DemanglerOutputDataReceived(object sender, DataReceivedEventArgs e) {

    }

    private static string DemangleName(string name) {
      return name;
    }

  }

  class StateIdFixer {

    // For race reporting, we emit a bunch of "state_id" attributes.
    // It is important that these are not duplicated.  However,
    // loop unrolling duplicates them.  This class is responsible for
    // fixing things up.  It is not a particularly elegant solution.

    private int CheckStateCounter = 0;
    private int LoopHeadStateCounter = 0;
    private int CallReturnStateCounter = 0;

    internal void FixStateIds(Program Program) {

      Debug.Assert(CommandLineOptions.Clo.LoopUnrollCount != -1);

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
          if(StateName.Contains("check_state")) {
            // It is necessary to clone the assume and call command, because after loop unrolling
            // there is aliasing between blocks of different loop iterations
            newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, ResetCheckStateId(a.Attributes, "captureState")));

            #region Skip on to the next call, adding all intervening commands to the new command list
            CallCmd c;
            do {
              i++;
              Debug.Assert(i < b.Cmds.Count());
              c = b.Cmds[i] as CallCmd;
              if(c == null) {
                newCmds.Add(b.Cmds[i]);
              }
            } while(c == null);
            Debug.Assert(c != null);
            #endregion

            Debug.Assert(QKeyValue.FindStringAttribute(c.Attributes, "state_id") != null);
            var newCall = new CallCmd(Token.NoToken, c.callee, c.Ins, c.Outs, ResetCheckStateId(c.Attributes, "state_id"));
            newCall.Proc = c.Proc;
            newCmds.Add(newCall);
            CheckStateCounter++;
          } else if(StateName.Contains("call_return_state")) {
            newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, ResetStateId(a.Attributes, "call_return_state_", CallReturnStateCounter)));
            CallReturnStateCounter++;
          } else {
            Debug.Assert(StateName.Contains("loop_head_state"));
            newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, ResetStateId(a.Attributes, "loop_head_state_", LoopHeadStateCounter)));
            LoopHeadStateCounter++;
          }
        }
        else {
          newCmds.Add(b.Cmds[i]);
        }
      }
      b.Cmds = newCmds;
      return b;
    }

    private QKeyValue ResetCheckStateId(QKeyValue Attributes, string Key) {
      // This demands special treatment.
      // Returns attributes identical to Attributes, but:
      // - reversed (for ease of implementation; should not matter)
      // - with the value for Key replaced by "check_state_X" where X is the counter field
      Debug.Assert(QKeyValue.FindStringAttribute(Attributes, Key) != null);
      QKeyValue result = null;
      while (Attributes != null) {
        if (Attributes.Key.Equals(Key)) {
          result = new QKeyValue(Token.NoToken, Attributes.Key, new List<object>() { "check_state_" + CheckStateCounter }, result);
        }
        else {
          result = new QKeyValue(Token.NoToken, Attributes.Key, Attributes.Params, result);
        }
        Attributes = Attributes.Next;
      }
      return result;
    }

    private QKeyValue ResetStateId(QKeyValue Attributes, string prefix, int counter) {
      // Returns attributes identical to Attributes, but:
      // - reversed (for ease of implementation; should not matter)
      // - with the value for "captureState" replaced by prefix_counter
      Debug.Assert(QKeyValue.FindStringAttribute(Attributes, "captureState") != null);
      QKeyValue result = null;
      while (Attributes != null) {
        if (Attributes.Key.Equals("captureState")) {
          result = new QKeyValue(Token.NoToken, Attributes.Key, new List<object>() { prefix + counter }, result);
        }
        else {
          result = new QKeyValue(Token.NoToken, Attributes.Key, Attributes.Params, result);
        }
        Attributes = Attributes.Next;
      }
      return result;
    }


  }

}
