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


namespace GPUVerify {

  class GPUVerifyErrorReporter {

    enum RaceType {
      WW,
      RW,
      WR
    };

    enum ErrorMsgType {
      Error,
      Note,
      NoError
    };

    internal static void ReportCounterexample(Counterexample error) {
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
        else {
          ReportFailingAssert(AssertCex);
        }
      }
    }

    private static void ReportRace(CallCounterexample CallCex) {

      string thread1, thread2, group1, group2, arrName;

      Model ModelWithStates = CallCex.GetModelWithStates();

      uint byteOffset = GetOffsetInBytes(ExtractOffsetVar(CallCex), ModelWithStates, CallCex.FailingCall);

      GetThreadsAndGroupsFromModel(CallCex.Model, out thread1, out thread2, out group1, out group2, true);

      arrName = GetArrayName(CallCex.FailingRequires);

      Debug.Assert(arrName != null);

      if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_read")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, "WRITE", ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WR);
      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "read_write")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, "READ", ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RW);

      }
      else if (QKeyValue.FindBoolAttribute(CallCex.FailingRequires.Attributes, "write_write")) {
        CallCex.FailingRequires.Attributes = GetSourceLocInfo(CallCex, "WRITE", ModelWithStates);
        ReportRace(CallCex.FailingCall, CallCex.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WW);
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
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine() + "\n", 2));


      ErrorWriteLine(locinfo2, access1 + " by thread " + thread1 + " group " + group1, ErrorMsgType.NoError);
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine() + "\n", 2));
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
      var element = state.TryGet(OffsetVar.Name);

      Debug.Assert(element is Model.BitVector);
      var BitVectorElement = ((Model.BitVector)element);

      uint elemOffset = Convert.ToUInt32(BitVectorElement.Numeral);

      Debug.Assert(OffsetVar.Attributes != null);
      uint elemWidth = (uint)QKeyValue.FindIntAttribute(OffsetVar.Attributes, "elem_width", int.MaxValue);
      Debug.Assert(elemWidth != int.MaxValue);
      return (elemOffset * elemWidth) / 8;
    }

    private static Variable ExtractOffsetVar(CallCounterexample err) {
      // The offset variable name can be exactly reconstructed from the attributes of the requires clause
      string ArrayName = QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array");
      string AccessType;
      if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_write") ||
         QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_read")) {
        AccessType = "WRITE";
      }
      else {
        Debug.Assert(QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_write"));
        AccessType = "READ";
      }
      string OffsetVarName = "_" + AccessType + "_OFFSET_" + ArrayName + "$1";

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
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(sli.FetchCodeLine());
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

    private static void ReportEnsuresFailure(Absy node) {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "postcondition might not hold on all return paths", ErrorMsgType.Error);
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportBarrierDivergence(Absy node) {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode) {
      Console.WriteLine("");
      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      ErrorWriteLine(CallSLI.ToString(), "a precondition for this call might not hold", ErrorMsgType.Error);
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine(), 2));

      ErrorWriteLine(RequiresSLI.ToString(), "this is the precondition that might not hold", ErrorMsgType.Note);
      Microsoft.Boogie.GPUVerifyBoogieDriver.ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine(), 2));
    }

    private static void GetThreadsAndGroupsFromModel(Model model, out string thread1, out string thread2, out string group1, out string group2, bool withSpaces) {
      thread1 = GetThreadOne(model, withSpaces);
      thread2 = GetThreadTwo(model, withSpaces);
      group1 = GetGroupOne(model, withSpaces);
      group2 = GetGroupTwo(model, withSpaces);
    }

    private static string GetGroupTwo(Model model, bool withSpaces) {
      return "("
             + GetGidX2(model)
             + "," + (withSpaces ? " " : "")
             + GetGidY2(model)
             + "," + (withSpaces ? " " : "")
             + GetGidZ2(model)
             + ")";
    }

    private static int GetGidZ2(Model model) {
      return model.TryGetFunc("group_id_z$2").GetConstant().AsInt();
    }

    private static int GetGidY2(Model model) {
      return model.TryGetFunc("group_id_y$2").GetConstant().AsInt();
    }

    private static int GetGidX2(Model model) {
      return model.TryGetFunc("group_id_x$2").GetConstant().AsInt();
    }

    private static string GetGroupOne(Model model, bool withSpaces) {
      return "("
             + GetGidX1(model)
             + "," + (withSpaces ? " " : "")
             + GetGidY1(model)
             + "," + (withSpaces ? " " : "")
             + GetGidZ1(model)
             + ")";
    }

    private static int GetGidZ1(Model model) {
      return model.TryGetFunc("group_id_z$1").GetConstant().AsInt();
    }

    private static int GetGidY1(Model model) {
      return model.TryGetFunc("group_id_y$1").GetConstant().AsInt();
    }

    private static int GetGidX1(Model model) {
      return model.TryGetFunc("group_id_x$1").GetConstant().AsInt();
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

    static QKeyValue GetSourceLocInfo(CallCounterexample err, string AccessType, Model ModelWithStates) {
      string ArrayName = QKeyValue.FindStringAttribute(err.FailingRequires.Attributes, "array");
      Debug.Assert(ArrayName != null);
      string StateId = QKeyValue.FindStringAttribute(err.FailingCall.Attributes, "state_id");
      Debug.Assert(StateId != null);

      Model.CapturedState CapturedState = null;
      foreach (var s in ModelWithStates.States) {
        if (s.Name.Equals(StateId)) {
          CapturedState = s;
          break;
        }
      }
      Debug.Assert(CapturedState != null);

      string SourceVarName = "_" + AccessType + "_SOURCE_" + ArrayName + "$1";
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
  
  }


  class VariableFinderVisitor : StandardVisitor {

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
