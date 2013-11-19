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

    enum RaceType {
      WW,
      RW,
      WR,
      AR,
      AW,
      RA,
      WA
    };

    enum ErrorMsgType {
      Error,
      Note,
      NoError
    };


    private Program program;
    private Implementation impl;

    internal GPUVerifyErrorReporter(Program program, string implName) {
      this.program = program;
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

      if(impl.InParams.Count() == 0) {
        return;
      }

      Console.Error.WriteLine("Bitwise values of parameters of " + impl.Name.TrimStart(new char[] { '$' }) + ":");
      if (!error.ModelHasStatesAlready) {
        error.PopulateModelWithStates();
        error.ModelHasStatesAlready = true;
      }

      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(error.Model, out thread1, out thread2, out group1, out group2, false);
      foreach(var p in impl.InParams) {

        int id;
        string stripped = GVUtil.StripThreadIdentifier(p.Name, out id).TrimStart(new char[] { '$' });
        Console.Error.Write("  " + stripped + " = ");

        var func = error.Model.TryGetFunc(p.Name);
        if(func != null) {
          var val = func.GetConstant();
          if(val is Model.BitVector) {
            Console.Error.Write(((Model.BitVector)val).Numeral);
          } else if(val is Model.Uninterpreted) {
            Console.Error.Write("<irrelevant>");
          } else {
            Console.Error.Write("<unknown>");
          }
        } else {
          Console.Error.Write("<unknown>");
        }
        Console.Error.WriteLine(id == 1 ? " (thread " + thread1 + ", group " + group1 + ")" : 
                               (id == 2 ? " (thread " + thread2 + ", group " + group2 + ")" : ""));
      }
      Console.WriteLine();
    }

    private void ReportRace(CallCounterexample CallCex) {

      if (!CallCex.ModelHasStatesAlready) {
        CallCex.PopulateModelWithStates();
        CallCex.ModelHasStatesAlready = true;
      }
      Model ModelWithStates = CallCex.Model;

      uint byteOffset = GetOffsetInBytes(ExtractOffsetVar(CallCex), ModelWithStates, CallCex.FailingCall);

      string thread1, thread2, group1, group2;
      GetThreadsAndGroupsFromModel(CallCex.Model, out thread1, out thread2, out group1, out group2, true);

      string arrName = GetArrayName(CallCex.FailingRequires);
      Debug.Assert(arrName != null);

      if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_read")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WR);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_write")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RW);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_write")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WW);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_read")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.AR);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_write")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.AW);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_atomic")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RA);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_atomic")) {
        ReportRace(CallCex, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WA);
      }
    }

    private void ReportRace(CallCounterexample CallCex, string thread1, string thread2, string group1, string group2, string arrName, uint byteOffset, RaceType raceType) {
      string raceName, access1, access2;
      switch (raceType) {
        case RaceType.RW:
        raceName = "read-write";
        access1 = "Read";
        access2 = "Write";
        break;
        case RaceType.WR:
        raceName = "write-read";
        access1 = "Write";
        access2 = "Read";
        break;
        case RaceType.WW:
        raceName = "write-write";
        access1 = "Write";
        access2 = "Write";
        break;
        case RaceType.AR:
        raceName = "atomic-read";
        access1 = "Atomic";
        access2 = "Read";
        break;
        case RaceType.AW:
        raceName = "atomic-write";
        access1 = "Atomic";
        access2 = "Write";
        break;
        case RaceType.RA:
        raceName = "read-atomic";
        access1 = "Read";
        access2 = "Atomic";
        break;
        case RaceType.WA:
        raceName = "write-atomic";
        access1 = "Write";
        access2 = "Atomic";
        break;

        default:
        raceName = null;
        access1 = null;
        access2 = null;
        Debug.Assert(false, "ReportRace(): Reached default case in switch over raceType.");
        break;
      }


      SourceLocationInfo SourceInfoForSecondAccess = new SourceLocationInfo(GetAttributes(CallCex.FailingCall), CallCex.FailingCall.tok);
      HashSet<SourceLocationInfo> PossibleSourcesForFirstAccess = GetPossibleSourceLocationsForFirstAccessInRace(CallCex, arrName, access1);

      Console.Error.WriteLine();
      ErrorWriteLine(SourceInfoForSecondAccess.GetFile() + ":", "possible " + raceName + " race on ((char*)" + arrName + ")[" + byteOffset + "]:\n", ErrorMsgType.Error);

      Console.Error.WriteLine(access2 + " by thread " + thread2 + " in group " + group2 + ", " + SourceInfoForSecondAccess);
      GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(SourceInfoForSecondAccess.FetchCodeLine() + "\n", 2));

      Console.Error.Write(access1 + " by thread " + thread1 + " in group " + group1 + ", ");
      if(PossibleSourcesForFirstAccess.Count() == 1) {
        Console.Error.WriteLine(PossibleSourcesForFirstAccess.ToList()[0]);
        GVUtil.IO.ErrorWriteLine(TrimLeadingSpaces(PossibleSourcesForFirstAccess.ToList()[0].FetchCodeLine() + "\n", 2));
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

    private HashSet<SourceLocationInfo> GetPossibleSourceLocationsForFirstAccessInRace(CallCounterexample CallCex, string arrayName, string accessType)
    {
      string ACCESS_HAS_OCCURRED = "_" + accessType.ToUpper() + "_HAS_OCCURRED_$$" + arrayName + "$1";
      string ACCESS_OFFSET = "_" + accessType.ToUpper() + "_OFFSET_$$" + arrayName + "$1";

      Tuple<AssumeCmd, string> LastLogPosition = null;

      foreach (var b in CallCex.Trace)
      {
        foreach (var c in b.Cmds.OfType<AssumeCmd>())
        {
          string StateName = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
          if (StateName == null)
          {
            continue;
          }
          Model.CapturedState state = GetStateFromModel(StateName, CallCex.Model);
          if(state.TryGet(ACCESS_HAS_OCCURRED) is Model.Uninterpreted) {
            // This value has nothing to do with the reported error, so do not
            // analyse it further.
            continue;
          }

          Model.Boolean AHO_value = (Model.Boolean)state.TryGet(ACCESS_HAS_OCCURRED);
          Model.BitVector AO_value = (Model.BitVector)state.TryGet(ACCESS_OFFSET);
          if (!AHO_value.Value)
          {
            LastLogPosition = null;
          }
          else if (LastLogPosition == null || !AO_value.Numeral.Equals(LastLogPosition.Item2))
          {
            LastLogPosition = new Tuple<AssumeCmd, string>(c, AO_value.Numeral);
          }
        }
      }

      Debug.Assert(LastLogPosition != null);

      var LastStateName = QKeyValue.FindStringAttribute(LastLogPosition.Item1.Attributes, "captureState");

      if (LastStateName.Contains("loop_head_state"))
      {
        Program originalProgram = GVUtil.GetFreshProgram(CommandLineOptions.Clo.Files, true, false);
        Implementation originalImplementation = originalProgram.Implementations().Where(Item => Item.Name.Equals(impl.Name)).ToList()[0];
        var blockGraph = program.ProcessLoops(originalImplementation);
        Block header = null;
        foreach (var b in blockGraph.Headers)
        {
          foreach (var c in b.Cmds.OfType<AssumeCmd>())
          {
            var stateId = QKeyValue.FindStringAttribute(c.Attributes, "captureState");
            if (stateId != null && stateId.Equals(QKeyValue.FindStringAttribute(LastLogPosition.Item1.Attributes, "captureState")))
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
        return GetSourceLocationsFromBlocks(arrayName, accessType, LoopNodes);
      }
      else if(LastStateName.Contains("call_return_state")  ) {
        Program originalProgram = GVUtil.GetFreshProgram(CommandLineOptions.Clo.Files, true, false);
        var CallGraph = Program.BuildCallGraph(originalProgram);
        HashSet<Implementation> PossiblyInvokedProcedures = new HashSet<Implementation>();
        string CalleeName = QKeyValue.FindStringAttribute(LastLogPosition.Item1.Attributes, "procedureName");
        Debug.Assert(CalleeName != null);
        PossiblyInvokedProcedures.Add(originalProgram.Implementations().Where(Item => Item.Name.Equals(CalleeName)).ToList()[0]);
        bool changed = true;
        while(changed) {
          changed = false;
          foreach(var impl in PossiblyInvokedProcedures.ToList()) {
            foreach(var succ in CallGraph.Successors(impl)) {
              Console.WriteLine(succ.Name);
              if(!PossiblyInvokedProcedures.Contains(succ)) {
                changed = true;
                PossiblyInvokedProcedures.Add(succ);
              }
            }
          }
        }
        Console.WriteLine("Procedures possibly invoked from call to " + CalleeName + " are:");
        foreach(var impl in PossiblyInvokedProcedures) {
          Console.WriteLine("  " + impl.Name);
        }

        return GetSourceLocationsFromBlocks(arrayName, accessType, 
          new HashSet<Block>(PossiblyInvokedProcedures.Select(Item => Item.Blocks).
          SelectMany(Item => Item)));
      } else {
        Debug.Assert(LastStateName.Contains("check_state"));
        return new HashSet<SourceLocationInfo> { 
          new SourceLocationInfo(LastLogPosition.Item1.Attributes, LastLogPosition.Item1.tok)
        };
      }
    }

    private static HashSet<SourceLocationInfo> GetSourceLocationsFromBlocks(string arrayName, string accessType, HashSet<Block> LoopNodes)
    {
      HashSet<SourceLocationInfo> PossibleSources = new HashSet<SourceLocationInfo>();
      foreach (var c in LoopNodes.Select(Item => Item.Cmds).SelectMany(Item => Item).OfType<CallCmd>())
      {
        if (c.callee.Equals("_CHECK_" + accessType.ToUpper() + "_$$" + arrayName))
        {
          PossibleSources.Add(new SourceLocationInfo(c.Attributes, c.tok));
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
      Debug.Assert(state != null);
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

      string OffsetVarName = "_" + Access + "_OFFSET_" + ArrayName + "$1";

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
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "postcondition might not hold on all return paths", ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportBarrierDivergence(Absy node) {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      GVUtil.IO.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode) {
      Console.WriteLine("");
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

    private static string GetSourceLocFileName() {
      return GetFilenamePathPrefix() + GetFileName() + ".loc";
    }

    private static string GetFileName() {
      return Path.GetFileNameWithoutExtension(CommandLineOptions.Clo.Files[0]);
    }

    private static string GetFilenamePathPrefix() {
      string directoryName = Path.GetDirectoryName(CommandLineOptions.Clo.Files[0]);
      return ((!String.IsNullOrEmpty(directoryName) && directoryName != ".") ? (directoryName + Path.DirectorySeparatorChar) : "");
    }

    private static string GetCorrespondingThreadTwoName(string threadOneName) {
      return threadOneName.Replace("$1", "$2");
    }

    private static QKeyValue CreateSourceLocQKV(int line, int col, string fname, string dir) {
      QKeyValue dirkv = new QKeyValue(Token.NoToken, "dir", new List<object>(new object[] { dir }), null);
      QKeyValue fnamekv = new QKeyValue(Token.NoToken, "fname", new List<object>(new object[] { fname }), dirkv);
      QKeyValue colkv = new QKeyValue(Token.NoToken, "col", new List<object>(new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(col)) }), fnamekv);
      QKeyValue linekv = new QKeyValue(Token.NoToken, "line", new List<object>(new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(line)) }), colkv);
      return linekv;
    }

    public static void FixStateIds(Program Program) {
      new StateIdFixer().FixStateIds(Program);
    }

  }

  class StateIdFixer {

    // For race reporting, we emit a bunch of "state_id" attributes.
    // It is important that these are not duplicated.  However,
    // loop unrolling duplicates them.  This class is responsible for
    // fixing things up.  It is not a particularly elegant solution.

    private int CheckStateCounter = 0;
    private int LoopHeadStateCounter = 0;

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
          if(QKeyValue.FindStringAttribute(a.Attributes, "captureState").Contains("check_state")) {
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
          } else {
            Debug.Assert(QKeyValue.FindStringAttribute(a.Attributes, "captureState").Contains("loop_head_state"));
            newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, ResetLoopHeadStateId(a.Attributes)));
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

    private QKeyValue ResetLoopHeadStateId(QKeyValue Attributes) {
      // Returns attributes identical to Attributes, but:
      // - reversed (for ease of implementation; should not matter)
      // - with the value for "captureState" replaced by "loop_head_state_X" where X is the counter field
      Debug.Assert(QKeyValue.FindStringAttribute(Attributes, "captureState") != null);
      QKeyValue result = null;
      while (Attributes != null) {
        if (Attributes.Key.Equals("captureState")) {
          result = new QKeyValue(Token.NoToken, Attributes.Key, new List<object>() { "loop_head_state_" + LoopHeadStateCounter }, result);
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
