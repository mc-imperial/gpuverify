//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;
using Microsoft.Basetypes;
using System.Text.RegularExpressions;
using System.Diagnostics.Contracts;

namespace GPUVerify
{
  public class GPUVerifyErrorReporter
  {
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

    public static void ReportCounterexample(Counterexample error) {
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
        else {
          ReportFailingAssert(AssertCex);
        }
      }
    }

    private static void ReportRace(CallCounterexample CallCex) {

      string thread1, thread2, group1, group2, arrName;

      if (!CallCex.ModelHasStatesAlready) {
        CallCex.PopulateModelWithStates();
        CallCex.ModelHasStatesAlready = true;
      }
      Model ModelWithStates = CallCex.Model;

      uint byteOffset = GetOffsetInBytes(ExtractOffsetVar(CallCex), ModelWithStates, CallCex.FailingCall);

      GetThreadsAndGroupsFromModel(CallCex.Model, out thread1, out thread2, out group1, out group2, true);

      arrName = GetArrayName(CallCex.FailingRequires);

      Debug.Assert(arrName != null);

      if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_read")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.WRITE, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WR);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_write")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.READ, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RW);

      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_write")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.WRITE, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WW);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_read")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.ATOMIC, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.AR);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "atomic_write")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.ATOMIC, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.AW);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_atomic")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.READ, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RA);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_atomic")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, AccessType.WRITE, ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WA);
      }
    }

    private static void ReportRace(CallCmd callNode, Requires reqNode, string thread1, string thread2, string group1, string group2, string arrName, uint byteOffset, RaceType raceType) {
      Console.WriteLine("");
      string locinfo1 = null, locinfo2 = null, raceName, access1, access2;

      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      switch (raceType) {
        case RaceType.RW:
        raceName = "read-write";
        access1 = "read";
        access2 = "write";
        break;
        case RaceType.WR:
        raceName = "write-read";
        access1 = "write";
        access2 = "read";
        break;
        case RaceType.WW:
        raceName = "write-write";
        access1 = "write";
        access2 = "write";
        break;
        case RaceType.AR:
        raceName = "atomic-read";
        access1 = "atomic";
        access2 = "read";
        break;
        case RaceType.AW:
        raceName = "atomic-write";
        access1 = "atomic";
        access2 = "write";
        break;
        case RaceType.RA:
        raceName = "read-atomic";
        access1 = "read";
        access2 = "atomic";
        break;
        case RaceType.WA:
        raceName = "write-atomic";
        access1 = "write";
        access2 = "atomic";
        break;

        default:
        raceName = null;
        access1 = null;
        access2 = null;
        Debug.Assert(false, "ReportRace(): Reached default case in switch over raceType.");
        break;
      }

      ErrorWriteLine(CallSLI.GetFile() + ":", "possible " + raceName + " race on ((char*)" + arrName + ")[" + byteOffset + "]:\n", ErrorMsgType.Error);

      locinfo1 = CallSLI.ToString();
      locinfo2 = RequiresSLI.ToString();

      AddPadding(ref locinfo1, ref locinfo2);

      ErrorWriteLine(locinfo1, access2 + " by thread " + thread2 + " group " + group2, ErrorMsgType.NoError);
      GVUtil.ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine() + "\n", 2));


      ErrorWriteLine(locinfo2, access1 + " by thread " + thread1 + " group " + group1, ErrorMsgType.NoError);
      GVUtil.ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine() + "\n", 2));
    }

    private static uint GetOffsetInBytes(Variable OffsetVar, Model m, CallCmd FailingCall) {
      string StateName = QKeyValue.FindStringAttribute(FailingCall.Attributes, "state_id");
      Debug.Assert(StateName != null);
      Model.CapturedState state = null;
      foreach (var s in m.States) {
        if (s.Name.Equals(StateName)) {
          state = s;
          break;
        }
      }
      Debug.Assert(state != null);
      var element = state.TryGet(OffsetVar.Name) as Model.Number;
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
                     (relevantThread == 1 ? thread1 : thread2) + " group " + (relevantThread == 1 ? group1 : group2), ErrorMsgType.Error);
      GVUtil.ErrorWriteLine(sli.FetchCodeLine());
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

    private static void ReportEnsuresFailure(Absy node) {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "postcondition might not hold on all return paths", ErrorMsgType.Error);
      GVUtil.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportBarrierDivergence(Absy node) {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      GVUtil.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode) {
      Console.WriteLine("");
      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      ErrorWriteLine(CallSLI.ToString(), "a precondition for this call might not hold", ErrorMsgType.Error);
      GVUtil.ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine(), 2));

      ErrorWriteLine(RequiresSLI.ToString(), "this is the precondition that might not hold", ErrorMsgType.Note);
      GVUtil.ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine(), 2));
    }

    private static void GetThreadsAndGroupsFromModel(Model model, out string thread1, out string thread2, out string group1, out string group2, bool withSpaces) {
      thread1 = GetThreadOne(model, withSpaces);
      thread2 = GetThreadTwo(model, withSpaces);
      group1 = GetGroup(model, withSpaces, 1);
      group2 = GetGroup(model, withSpaces, 2);
    }

    private static string GetGroup(Model model, bool withSpaces, int thread) {
      return "("
        + GetGid(model, "x", thread)
          + "," + (withSpaces ? " " : "")
          + GetGid(model, "y", thread)
          + "," + (withSpaces ? " " : "")
          + GetGid(model, "z", thread)
          + ")";
    }

    private static int GetGid(Model model, string dimension, int thread) {
      string name = "group_id_" + dimension;
      if(!((GPUVerifyKernelAnalyserCommandLineOptions)CommandLineOptions.Clo).OnlyIntraGroupRaceChecking) {
        name += "$" + thread;
      }

      return model.TryGetFunc(name).GetConstant().AsInt();
    }

    private static string GetThreadTwo(Model model, bool withSpaces) {
      return "("
        + GetLidX2(model)
          + "," + (withSpaces ? " " : "")
          + GetLidY2(model)
          + "," + (withSpaces ? " " : "")
          + GetLidZ2(model)
          + ")";
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
      return "("
        + model.TryGetFunc("local_id_x$1").GetConstant().AsInt()
          + "," + (withSpaces ? " " : "")
          + model.TryGetFunc("local_id_y$1").GetConstant().AsInt()
          + "," + (withSpaces ? " " : "")
          + model.TryGetFunc("local_id_z$1").GetConstant().AsInt()
          + ")";
    }

    private static string GetArrayName(Requires requires) {
      string arrName = QKeyValue.FindStringAttribute(requires.Attributes, "array");
      Debug.Assert(arrName != null);
      Debug.Assert(arrName.StartsWith("$$"));
      return arrName.Substring("$$".Length);
    }

    private static void AddPadding(ref string string1, ref string string2) {
      if (string1.Length < string2.Length) {
        for (int i = (string2.Length - string1.Length); i > 0; --i) {
          string1 += " ";
        }
      }
      else {
        for (int i = (string1.Length - string2.Length); i > 0; --i) {
          string2 += " ";
        }
      }
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

    static QKeyValue GetSourceLocInfo(CallCounterexample err, AccessType Access, Model ModelWithStates) {
      string ArrayName = QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array");
      Debug.Assert(ArrayName != null);
      string StateId = QKeyValue.FindStringAttribute(err.FailingCall.Attributes, "state_id");
      Debug.Assert(StateId != null);

      if ((CommandLineOptions.Clo as GPUVerifyKernelAnalyserCommandLineOptions).NoSourceLocInfer) {
        return CreateSourceLocQKV(0,0,GetFileName(),GetFilenamePathPrefix());
      }

      Model.CapturedState CapturedState = null;
      foreach (var s in ModelWithStates.States) {
        if (s.Name.Equals(StateId)) {
          CapturedState = s;
          break;
        }
      }
      Debug.Assert(CapturedState != null);

      string SourceVarName = "_" + Access + "_SOURCE_" + ArrayName + "$1";
      int SourceValue = CapturedState.TryGet(SourceVarName).AsInt();

      try {
        // TODO: Make lines in .loc file be indexed from 1 for consistency.
        string fileLine = SourceLocationInfo.FetchCodeLine(GetSourceLocFileName(), SourceValue + 1);
        string[] slocTokens = Regex.Split(fileLine, "#");
        return CreateSourceLocQKV(
          System.Convert.ToInt32(slocTokens[0]),
          System.Convert.ToInt32(slocTokens[1]),
          slocTokens[2],
          slocTokens[3]);
      }
      catch (Exception) {
        return null;
      }
    }

    public static void FixStateIds(Program Program) {
      new StateIdFixer().FixStateIds(Program);
    }
  }

  class StateIdFixer
  {
    // For race reporting, we emit a bunch of "state_id" attributes.
    // It is important that these are not duplicated.  However,
    // loop unrolling duplicates them.  This class is responsible for
    // fixing things up.  It is not a particularly elegant solution.

    private int counter = 0;

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
          // It is necessary to clone the assume and call command, because after loop unrolling
          // there is aliasing between blocks of different loop iterations
          newCmds.Add(new AssumeCmd(Token.NoToken, a.Expr, ResetStateId(a.Attributes, "captureState")));

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
          var newCall = new CallCmd(Token.NoToken, c.callee, c.Ins, c.Outs, ResetStateId(c.Attributes, "state_id"));
          newCall.Proc = c.Proc;
          newCmds.Add(newCall);
          counter++;
        }
        else {
          newCmds.Add(b.Cmds[i]);
        }
      }
      b.Cmds = newCmds;
      return b;
    }

    private QKeyValue ResetStateId(QKeyValue Attributes, string Key) {
      // Returns attributes identical to Attributes, but:
      // - reversed (for ease of implementation; should not matter)
      // - with the value for Key replaced by "check_state_X" where X is the counter field
      Debug.Assert(QKeyValue.FindStringAttribute(Attributes, Key) != null);
      QKeyValue result = null;
      while (Attributes != null) {
        if (Attributes.Key.Equals(Key)) {
          result = new QKeyValue(Token.NoToken, Attributes.Key, new List<object>() { "check_state_" + counter }, result);
        }
        else {
          result = new QKeyValue(Token.NoToken, Attributes.Key, Attributes.Params, result);
        }
        Attributes = Attributes.Next;
      }
      return result;
    }

  }

  class VariableFinderVisitor : StandardVisitor
  {
    private string VarName;
    private Variable Variable = null;

    internal VariableFinderVisitor(string VarName) {
      this.VarName = VarName;
    }

    public override Variable VisitVariable(Variable node) {
      if (node.Name.Equals(VarName)) {
        Variable = node;
      }
      return base.VisitVariable(node);
    }

    internal Variable GetVariable() {
      return Variable;
    }
  }
}
