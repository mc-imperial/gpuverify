//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Diagnostics.Contracts;
using Microsoft.Boogie;
using Microsoft.Basetypes;

namespace GPUVerify {

  class RaceInstrumenter : IRaceInstrumenter {
    protected GPUVerifier verifier;

    private QKeyValue SourceLocationAttributes = null;

    private int CurrStmtNo = 1;

    private int CheckStateCounter = 0;

    private Dictionary<string, List<int>> ReadAccessSourceLocations = new Dictionary<string, List<int>>();
    private Dictionary<string, List<int>> WriteAccessSourceLocations = new Dictionary<string, List<int>>();
    private Dictionary<string, List<int>> AtomicAccessSourceLocations = new Dictionary<string, List<int>>();

    public IKernelArrayInfo NonLocalStateToCheck;

    private Dictionary<string, Procedure> RaceCheckingProcedures = new Dictionary<string, Procedure>();

    public RaceInstrumenter(GPUVerifier verifier) {
      this.verifier = verifier;
      NonLocalStateToCheck = new KernelArrayInfoLists();
      foreach (Variable v in verifier.KernelArrayInfo.getGlobalArrays()) {
        NonLocalStateToCheck.getGlobalArrays().Add(v);
      }
      foreach (Variable v in verifier.KernelArrayInfo.getGroupSharedArrays()) {
        NonLocalStateToCheck.getGroupSharedArrays().Add(v);
      }
    }

    private void AddNoAccessCandidateInvariants(IRegion region, Variable v) {

      // Reasoning: if READ_HAS_OCCURRED_v is not in the modifies set for the
      // loop then there is no point adding an invariant
      //
      // If READ_HAS_OCCURRED_v is in the modifies set, but the loop does not
      // contain a barrier, then it is almost certain that a read CAN be
      // pending at the loop head, so the invariant will not hold
      //
      // If there is a barrier in the loop body then READ_HAS_OCCURRED_v will
      // be in the modifies set, but there may not be a live read at the loop
      // head, so it is worth adding the loop invariant candidate.
      //
      // The same reasoning applies for WRITE

      if (verifier.ContainsBarrierCall(region)) {
        foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
        {
          if (verifier.ContainsNamedVariable(
              LoopInvariantGenerator.GetModifiedVariables(region), GPUVerifier.MakeAccessHasOccurredVariableName(v.Name, kind))) {
            AddNoAccessCandidateInvariant(region, v, kind);
          }
        }
      }
    }

    private void AddNoAccessCandidateRequires(Procedure Proc, Variable v) {
      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
        AddNoAccessCandidateRequires(Proc, v, kind, "1");
    }

    private void AddNoAccessCandidateEnsures(Procedure Proc, Variable v) {
      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
        AddNoAccessCandidateEnsures(Proc, v, kind, "1");
    }

    private void AddNoAccessCandidateInvariant(IRegion region, Variable v, string Access) {
      Expr candidate = NoAccessExpr(v, Access, "1");
      verifier.AddCandidateInvariant(region, candidate, "no " + Access.ToLower(), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    public void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region) {
      List<Expr> offsetPredicatesRead = new List<Expr>();
      List<Expr> offsetPredicatesWrite = new List<Expr>();
      List<Expr> offsetPredicatesAtomic = new List<Expr>();

      foreach (Variable v in NonLocalStateToCheck.getAllNonLocalArrays()) {
        AddNoAccessCandidateInvariants(region, v);
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, "READ");
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, "WRITE");
        AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(impl, region, v, "ATOMIC");
        offsetPredicatesRead = CollectOffsetPredicates(impl, region, v, "READ");
        offsetPredicatesWrite = CollectOffsetPredicates(impl, region, v, "WRITE");
        offsetPredicatesAtomic = CollectOffsetPredicates(impl, region, v, "ATOMIC");
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "READ", offsetPredicatesRead);
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "WRITE", offsetPredicatesWrite);
        AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "ATOMIC", offsetPredicatesAtomic);
        if (CommandLineOptions.InferSourceLocation) {
          AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "READ", new List<Expr>(
            offsetPredicatesRead.Zip(CollectSourceLocPredicates(region, v, "READ"), Expr.And)), true);
          AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "WRITE", new List<Expr>(
            offsetPredicatesWrite.Zip(CollectSourceLocPredicates(region, v, "WRITE"), Expr.And)), true);
          AddOffsetsSatisfyPredicatesCandidateInvariant(region, v, "ATOMIC", new List<Expr>(
            offsetPredicatesAtomic.Zip(CollectSourceLocPredicates(region, v, "ATOMIC"), Expr.And)), true);
        }
      }
    }

    private bool DoesNotReferTo(Expr expr, string v) {
      FindReferencesToNamedVariableVisitor visitor = new FindReferencesToNamedVariableVisitor(v);
      visitor.VisitExpr(expr);
      return !visitor.found;
    }

    private int ParameterOffsetForSource() {
      if (CommandLineOptions.NoBenign) {
        return 2;
      }
      else {
        return 3;
      }
    }

    private List<Expr> CollectSourceLocPredicates(IRegion region, Variable v, string accessType) {
      var sourceVar = verifier.FindOrCreateSourceVariable(v.Name, accessType);
      var sourceExpr = new IdentifierExpr(Token.NoToken, sourceVar);
      var sourcePreds = new List<Expr>();

      foreach (Cmd c in region.Cmds()) {
        if (c is CallCmd) {
          CallCmd call = c as CallCmd;
          if (call.callee == "_LOG_" + accessType + "_" + v.Name) {
            sourcePreds.Add(Expr.Eq(sourceExpr, call.Ins[ParameterOffsetForSource()]));
          }
        }
      }

      return sourcePreds;
    }
    private List<Expr> CollectOffsetPredicates(Implementation impl, IRegion region, Variable v, string accessType) {
      var offsetVar = new VariableDualiser(1, null, null).VisitVariable(verifier.MakeOffsetVariable(v.Name, accessType));
      var offsetExpr = new IdentifierExpr(Token.NoToken, offsetVar);
      var offsetPreds = new List<Expr>();

      foreach (var offset in GetOffsetsAccessed(region, v, accessType)) {
        bool isConstant;
        var def = verifier.varDefAnalyses[impl].SubstDefinitions(offset, impl.Name, out isConstant);
        if (def == null)
          continue;
        if (isConstant) {
          offsetPreds.Add(Expr.Eq(offsetExpr, def));
        }
        else {
          var sc = StrideConstraint.FromExpr(verifier, impl, def);
          var pred = sc.MaybeBuildPredicate(verifier, offsetExpr);
          if (pred != null)
            offsetPreds.Add(pred);
        }
      }

      return offsetPreds;
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateInvariants(Implementation impl, IRegion region, Variable v, string accessType) {
      KeyValuePair<IdentifierExpr, Expr> iLessThanC = GetILessThanC(region.Guard());
      if (iLessThanC.Key != null) {
        foreach (Expr e in GetOffsetsAccessed(region, v, accessType)) {
          if (HasFormIPlusLocalIdTimesC(e, iLessThanC, impl)) {
            AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(region, v, iLessThanC.Value, accessType);
            break;
          }
        }

        foreach (Expr e in GetOffsetsAccessed(region, v, accessType)) {
          if (HasFormIPlusGlobalIdTimesC(e, iLessThanC, impl)) {
            AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(region, v, iLessThanC.Value, accessType);
            break;
          }
        }

      }


    }

    private bool HasFormIPlusLocalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl) {
      if (!(e is NAryExpr)) {
        return false;
      }

      NAryExpr nary = e as NAryExpr;

      if (!nary.Fun.FunctionName.Equals("BV32_ADD")) {
        return false;
      }

      return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
          IsLocalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
          (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
          IsLocalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
    }

    private bool IsLocalIdTimesConstant(Expr maybeLocalIdTimesConstant, Expr constant, Implementation impl) {
      if (!(maybeLocalIdTimesConstant is NAryExpr)) {
        return false;
      }
      NAryExpr nary = maybeLocalIdTimesConstant as NAryExpr;
      if (!nary.Fun.FunctionName.Equals("BV32_MUL")) {
        return false;
      }

      return
          (SameConstant(nary.Args[0], constant) && verifier.IsLocalId(nary.Args[1], 0, impl)) ||
          (SameConstant(nary.Args[1], constant) && verifier.IsLocalId(nary.Args[0], 0, impl));
    }


    private bool HasFormIPlusGlobalIdTimesC(Expr e, KeyValuePair<IdentifierExpr, Expr> iLessThanC, Implementation impl) {
      if (!(e is NAryExpr)) {
        return false;
      }

      NAryExpr nary = e as NAryExpr;

      if (!nary.Fun.FunctionName.Equals("BV32_ADD")) {
        return false;
      }

      return (SameIdentifierExpression(nary.Args[0], iLessThanC.Key) &&
          IsGlobalIdTimesConstant(nary.Args[1], iLessThanC.Value, impl)) ||
          (SameIdentifierExpression(nary.Args[1], iLessThanC.Key) &&
          IsGlobalIdTimesConstant(nary.Args[0], iLessThanC.Value, impl));
    }

    private bool IsGlobalIdTimesConstant(Expr maybeGlobalIdTimesConstant, Expr constant, Implementation impl) {
      if (!(maybeGlobalIdTimesConstant is NAryExpr)) {
        return false;
      }
      NAryExpr nary = maybeGlobalIdTimesConstant as NAryExpr;
      if (!nary.Fun.FunctionName.Equals("BV32_MUL")) {
        return false;
      }

      return
          (SameConstant(nary.Args[0], constant) && verifier.IsGlobalId(nary.Args[1], 0, impl)) ||
          (SameConstant(nary.Args[1], constant) && verifier.IsGlobalId(nary.Args[0], 0, impl));
    }


    private bool SameConstant(Expr expr, Expr constant) {
      if (constant is IdentifierExpr) {
        IdentifierExpr identifierExpr = constant as IdentifierExpr;
        Debug.Assert(identifierExpr.Decl is Constant);
        return expr is IdentifierExpr && (expr as IdentifierExpr).Decl is Constant && (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Decl.Name);
      }
      else {
        Debug.Assert(constant is LiteralExpr);
        LiteralExpr literalExpr = constant as LiteralExpr;
        if (!(expr is LiteralExpr)) {
          return false;
        }
        if (!(literalExpr.Val is BvConst) || !((expr as LiteralExpr).Val is BvConst)) {
          return false;
        }

        return (literalExpr.Val as BvConst).Value.ToInt == ((expr as LiteralExpr).Val as BvConst).Value.ToInt;
      }
    }

    private bool SameIdentifierExpression(Expr expr, IdentifierExpr identifierExpr) {
      if (!(expr is IdentifierExpr)) {
        return false;
      }
      return (expr as IdentifierExpr).Decl.Name.Equals(identifierExpr.Name);
    }

    private KeyValuePair<IdentifierExpr, Expr> GetILessThanC(Expr expr) {

      if (expr is NAryExpr && (expr as NAryExpr).Fun.FunctionName.Equals("bv32_to_bool")) {
        expr = (expr as NAryExpr).Args[0];
      }

      if (!(expr is NAryExpr)) {
        return new KeyValuePair<IdentifierExpr, Expr>(null, null);
      }

      NAryExpr nary = expr as NAryExpr;

      if (!(nary.Fun.FunctionName.Equals("BV32_C_LT") || nary.Fun.FunctionName.Equals("BV32_LT"))) {
        return new KeyValuePair<IdentifierExpr, Expr>(null, null);
      }

      if (!(nary.Args[0] is IdentifierExpr)) {
        return new KeyValuePair<IdentifierExpr, Expr>(null, null);
      }

      if (!IsConstant(nary.Args[1])) {
        return new KeyValuePair<IdentifierExpr, Expr>(null, null);
      }

      return new KeyValuePair<IdentifierExpr, Expr>(nary.Args[0] as IdentifierExpr, nary.Args[1]);

    }

    private static bool IsConstant(Expr e) {
      return ((e is IdentifierExpr && (e as IdentifierExpr).Decl is Constant) || e is LiteralExpr);
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateRequires(Procedure Proc, Variable v) {
      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
        AddAccessedOffsetIsThreadLocalIdCandidateRequires(Proc, v, kind, 1);
    }

    private void AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(Procedure Proc, Variable v) {
      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
        AddAccessedOffsetIsThreadLocalIdCandidateEnsures(Proc, v, kind, 1);
    }

    public void AddKernelPrecondition() {
      foreach (Variable v in NonLocalStateToCheck.getAllNonLocalArrays()) {
        AddRequiresNoPendingAccess(v);
        AddRequiresSourceAccessZero(v);
      }
    }

    public void AddRaceCheckingInstrumentation() {

      foreach (Declaration d in verifier.Program.TopLevelDeclarations) {
        if (d is Implementation) {
          AddRaceCheckCalls(d as Implementation);
        }
      }

    }

    private void AddRaceCheckingDecsAndProcsForVar(Variable v) {
      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
      {
        AddLogRaceDeclarations(v, kind);
        AddLogAccessProcedure(v, kind);
        AddCheckAccessProcedure(v, kind);
      }
    }

    private StmtList AddRaceCheckCalls(StmtList stmtList) {
      Contract.Requires(stmtList != null);

      StmtList result = new StmtList(new List<BigBlock>(), stmtList.EndCurly);

      foreach (BigBlock bodyBlock in stmtList.BigBlocks) {
        result.BigBlocks.Add(AddRaceCheckCalls(bodyBlock));
      }
      return result;
    }

    private Block AddRaceCheckCalls(Block b) {
      b.Cmds = AddRaceCheckCalls(b.Cmds);
      return b;
    }

    private void AddRaceCheckCalls(Implementation impl) {
      impl.Blocks = impl.Blocks.Select(AddRaceCheckCalls).ToList();
    }

    private List<Cmd> AddRaceCheckCalls(List<Cmd> cs) {
      var result = new List<Cmd>();
      foreach (Cmd c in cs) {

        if (c is AssertCmd) {
          AssertCmd assertion = c as AssertCmd;
          if (QKeyValue.FindBoolAttribute(assertion.Attributes, "sourceloc")) {
            SourceLocationAttributes = assertion.Attributes;
            // Remove source location assertions
            continue;
          }
        }

        if (c is CallCmd) {
          CallCmd call = c as CallCmd;
          if (QKeyValue.FindBoolAttribute(call.Attributes,"atomic"))
          {
            AddLogAndCheckCalls(result,new AccessRecord((call.Ins[0] as IdentifierExpr).Decl,call.Ins[1]),"ATOMIC",null);
            (result[result.Count() - 1] as CallCmd).Attributes = new QKeyValue(Token.NoToken, "atomic_function", new List<object> (new object[] { QKeyValue.FindStringAttribute(call.Attributes, "atomic_function") }), (result[result.Count() - 1] as CallCmd).Attributes);
            result.Add(new HavocCmd(Token.NoToken, new List<IdentifierExpr>(call.Outs.ToArray()))); // TODO: check this is right
            continue;
          }
        }

        result.Add(c);

        if (c is CallCmd) {
          CallCmd call = c as CallCmd;
          if (verifier.GetImplementation(call.callee) == null) {
            // This procedure has no body, so if it can modify race checking
            // variables then we must regard it's source location as a source
            // location where an access occurs
            Procedure proc = verifier.GetProcedure(call.callee);
            HashSet<Variable> AccessSourceVariablesModifies = new HashSet<Variable>();
            foreach (IdentifierExpr m in proc.Modifies) {
              foreach (var v in verifier.KernelArrayInfo.getAllNonLocalArrays()) {
                if(m.Name.Equals(GPUVerifier.MakeAccessHasOccurredVariableName(v.Name, "READ")) ||
                   m.Name.Equals(GPUVerifier.MakeOffsetVariableName(v.Name, "READ"))) {
                  AddToAccessSourceLocations("READ", v.Name);
                  AccessSourceVariablesModifies.Add(verifier.MakeSourceVariable(v.Name, "READ"));
                } else if(m.Name.Equals(GPUVerifier.MakeAccessHasOccurredVariableName(v.Name, "WRITE")) ||
                   m.Name.Equals(GPUVerifier.MakeOffsetVariableName(v.Name, "WRITE"))) {
                     AddToAccessSourceLocations("WRITE", v.Name);
                     AccessSourceVariablesModifies.Add(verifier.MakeSourceVariable(v.Name, "WRITE"));
                } else if(m.Name.Equals(GPUVerifier.MakeAccessHasOccurredVariableName(v.Name, "ATOMIC")) ||
                   m.Name.Equals(GPUVerifier.MakeOffsetVariableName(v.Name, "ATOMIC"))) {
                     AddToAccessSourceLocations("ATOMIC", v.Name);
                     AccessSourceVariablesModifies.Add(verifier.MakeSourceVariable(v.Name, "ATOMIC"));
                }
              }
            }

            if (AccessSourceVariablesModifies.Count > 0) {
              SourceLocationAttributes = call.Attributes;
              TryWriteSourceLocToFile();
              CurrStmtNo++;
              foreach (var v in AccessSourceVariablesModifies) {
                proc.Modifies.Add(new IdentifierExpr(Token.NoToken, v));
              }
            }
          }
        }

        if (c is AssignCmd) {
          AssignCmd assign = c as AssignCmd;

          ReadCollector rc = new ReadCollector(NonLocalStateToCheck);
          foreach (var rhs in assign.Rhss)
            rc.Visit(rhs);
          if (rc.accesses.Count > 0) {
            foreach (AccessRecord ar in rc.accesses) {
              AddLogAndCheckCalls(result, ar, "READ", null);
            }
          }

          foreach (var LhsRhs in assign.Lhss.Zip(assign.Rhss)) {
            WriteCollector wc = new WriteCollector(NonLocalStateToCheck);
            wc.Visit(LhsRhs.Item1);
            if (wc.GetAccess() != null) {
              AccessRecord ar = wc.GetAccess();
              AddLogAndCheckCalls(result, ar, "WRITE", LhsRhs.Item2);
            }
          }
        }
      }
      return result;
    }

    private void AddLogAndCheckCalls(List<Cmd> result, AccessRecord ar, string Access, Expr Value) {
      result.Add(MakeLogCall(ar, Access, Value));
      if (!CommandLineOptions.OnlyLog) {
        result.Add(MakeCheckCall(result, ar, Access, Value));
      }
      AddToAccessSourceLocations(Access, ar.v.Name);
      TryWriteSourceLocToFile();
      CurrStmtNo++;
    }

    private CallCmd MakeCheckCall(List<Cmd> result, AccessRecord ar, string Access, Expr Value) {
      List<Expr> inParamsChk = new List<Expr>();
      inParamsChk.Add(ar.Index);
      MaybeAddValueParameter(inParamsChk, ar, Value);
      Procedure checkProcedure = GetRaceCheckingProcedure(Token.NoToken, "_CHECK_" + Access + "_" + ar.v.Name);
      verifier.OnlyThread2.Add(checkProcedure.Name);
      string CheckState = "check_state_" + CheckStateCounter;
      CheckStateCounter++;
      AssumeCmd captureStateAssume = new AssumeCmd(Token.NoToken, Expr.True);
      captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
        "captureState", new List<object>() { CheckState }, null);
      captureStateAssume.Attributes = new QKeyValue(Token.NoToken,
        "do_not_predicate", new List<object>() { }, captureStateAssume.Attributes);
      result.Add(captureStateAssume);
      CallCmd checkAccessCallCmd = new CallCmd(Token.NoToken, checkProcedure.Name, inParamsChk, new List<IdentifierExpr>());
      checkAccessCallCmd.Proc = checkProcedure;
      checkAccessCallCmd.Attributes = SourceLocationAttributes;
      checkAccessCallCmd.Attributes = new QKeyValue(Token.NoToken, "state_id", new List<object>() { CheckState }, checkAccessCallCmd.Attributes);
      return checkAccessCallCmd;
    }

    private CallCmd MakeLogCall(AccessRecord ar, string Access, Expr Value) {
      List<Expr> inParamsLog = new List<Expr>();
      inParamsLog.Add(ar.Index);
      MaybeAddValueParameter(inParamsLog, ar, Value);
      inParamsLog.Add(verifier.IntRep.GetLiteral(CurrStmtNo, 32));
      Procedure logProcedure = GetRaceCheckingProcedure(Token.NoToken, "_LOG_" + Access + "_" + ar.v.Name);
      verifier.OnlyThread1.Add(logProcedure.Name);
      CallCmd logAccessCallCmd = new CallCmd(Token.NoToken, logProcedure.Name, inParamsLog, new List<IdentifierExpr>());
      logAccessCallCmd.Proc = logProcedure;
      logAccessCallCmd.Attributes = SourceLocationAttributes;
      return logAccessCallCmd;
    }

    private void MaybeAddValueParameter(List<Expr> parameters, AccessRecord ar, Expr Value) {
      if (!CommandLineOptions.NoBenign) {
        if (Value != null) {
          parameters.Add(Value);
        }
        else {
          Expr e = Expr.Select(new IdentifierExpr(Token.NoToken, ar.v), new Expr[] { ar.Index });
          e.Type = (ar.v.TypedIdent.Type as MapType).Result;
          parameters.Add(e);
        }
      }
    }

    private void TryWriteSourceLocToFile() {
      if (QKeyValue.FindStringAttribute(SourceLocationAttributes, "fname") != null) {
        writeSourceLocToFile(SourceLocationAttributes, GPUVerifier.GetSourceLocFileName());
      }
    }

    private void AddToAccessSourceLocations(string Access, string Key) {
      var SourceLocations = (Access == "WRITE" ? WriteAccessSourceLocations : ( Access == "READ" ? ReadAccessSourceLocations : AtomicAccessSourceLocations));
      if(!SourceLocations.ContainsKey(Key)) {
          SourceLocations.Add(Key, new List<int>());
      }
      if (!SourceLocations[Key].Contains(CurrStmtNo)) {
        SourceLocations[Key].Add(CurrStmtNo);
      }
    }

    private BigBlock AddRaceCheckCalls(BigBlock bb) {
      BigBlock result = new BigBlock(bb.tok, bb.LabelName, AddRaceCheckCalls(bb.simpleCmds), null, bb.tc);

      if (bb.ec is WhileCmd) {
        WhileCmd WhileCommand = bb.ec as WhileCmd;
        result.ec = new WhileCmd(WhileCommand.tok, WhileCommand.Guard,
                WhileCommand.Invariants, AddRaceCheckCalls(WhileCommand.Body));
      }
      else if (bb.ec is IfCmd) {
        IfCmd IfCommand = bb.ec as IfCmd;
        Debug.Assert(IfCommand.elseIf == null); // We don't handle else if yet
        result.ec = new IfCmd(IfCommand.tok, IfCommand.Guard, AddRaceCheckCalls(IfCommand.thn), IfCommand.elseIf, IfCommand.elseBlock != null ? AddRaceCheckCalls(IfCommand.elseBlock) : null);
      }
      else if (bb.ec is BreakCmd) {
        result.ec = bb.ec;
      }
      else {
        Debug.Assert(bb.ec == null);
      }

      return result;
    }

    private Procedure GetRaceCheckingProcedure(IToken tok, string name) {
      if (RaceCheckingProcedures.ContainsKey(name)) {
        return RaceCheckingProcedures[name];
      }
      Procedure newProcedure = new Procedure(tok, name, new List<TypeVariable>(), new List<Variable>(), new List<Variable>(), new List<Requires>(), new List<IdentifierExpr>(), new List<Ensures>());
      RaceCheckingProcedures[name] = newProcedure;
      return newProcedure;
    }


    public BigBlock MakeResetReadWriteSetStatements(Variable v, Expr ResetCondition) {
      BigBlock result = new BigBlock(Token.NoToken, null, new List<Cmd>(), null, null);

      foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
      {
        Expr ResetAssumeGuard = Expr.Imp(ResetCondition, 
          Expr.Not(new IdentifierExpr(Token.NoToken,
            new VariableDualiser(1, null, null).VisitVariable(
              GPUVerifier.MakeAccessHasOccurredVariable(v.Name, kind)))));

        if (verifier.KernelArrayInfo.getGlobalArrays().Contains(v))
          ResetAssumeGuard = Expr.Imp(GPUVerifier.ThreadsInSameGroup(), ResetAssumeGuard);

        result.simpleCmds.Add(new AssumeCmd(Token.NoToken, ResetAssumeGuard));
      }
      return result;
    }

    protected Procedure MakeLogAccessProcedureHeader(Variable v, string Access) {
      List<Variable> inParams = new List<Variable>();

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable SourceParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_source", mt.Arguments[0]));
      Debug.Assert(!(mt.Result is MapType));

      inParams.Add(VariableForThread(1, PredicateParameter));
      inParams.Add(VariableForThread(1, OffsetParameter));
      if(!CommandLineOptions.NoBenign) {
        inParams.Add(VariableForThread(1, ValueParameter));
      }
      inParams.Add(VariableForThread(1, SourceParameter));

      string LogProcedureName = "_LOG_" + Access + "_" + v.Name;

      Procedure result = GetRaceCheckingProcedure(v.tok, LogProcedureName);

      result.InParams = inParams;

      GPUVerifier.AddInlineAttribute(result);

      return result;
    }

    protected Procedure MakeCheckAccessProcedureHeader(Variable v, string Access) {
      List<Variable> inParams = new List<Variable>();

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Debug.Assert(!(mt.Result is MapType));

      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

      inParams.Add(VariableForThread(2, PredicateParameter));
      inParams.Add(VariableForThread(2, OffsetParameter));
      if (!CommandLineOptions.NoBenign) {
        inParams.Add(VariableForThread(2, ValueParameter));
      }

      string CheckProcedureName = "_CHECK_" + Access + "_" + v.Name;

      Procedure result = GetRaceCheckingProcedure(v.tok, CheckProcedureName);

      result.InParams = inParams;

      return result;
    }

    public void AddRaceCheckingCandidateRequires(Procedure Proc) {
      foreach (Variable v in NonLocalStateToCheck.getAllNonLocalArrays()) {
        AddNoAccessCandidateRequires(Proc, v);
        AddReadOrWrittenOffsetIsThreadIdCandidateRequires(Proc, v);
      }
    }

    public void AddRaceCheckingCandidateEnsures(Procedure Proc) {
      foreach (Variable v in NonLocalStateToCheck.getAllNonLocalArrays()) {
        AddNoAccessCandidateEnsures(Proc, v);
        AddReadOrWrittenOffsetIsThreadIdCandidateEnsures(Proc, v);
      }
    }

    private void AddNoAccessCandidateRequires(Procedure Proc, Variable v, string Access, string OneOrTwo) {
      verifier.AddCandidateRequires(Proc, NoAccessExpr(v, Access, OneOrTwo), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    private void AddNoAccessCandidateEnsures(Procedure Proc, Variable v, string Access, string OneOrTwo) {
      verifier.AddCandidateEnsures(Proc, NoAccessExpr(v, Access, OneOrTwo), InferenceStages.NO_READ_WRITE_CANDIDATE_STAGE);
    }

    private HashSet<Expr> GetOffsetsAccessed(IRegion region, Variable v, string Access) {
      HashSet<Expr> result = new HashSet<Expr>();

      foreach (Cmd c in region.Cmds()) {
        if (c is CallCmd) {
          CallCmd call = c as CallCmd;

          if (call.callee == "_LOG_" + Access + "_" + v.Name) {
            // Ins[0] is thread 1's predicate,
            // Ins[1] is the offset to be read
            // If Ins[1] has the form BV32_ADD(offset#construct...(P), offset),
            // we are looking for the second parameter to this BV32_ADD
            Expr offset = call.Ins[1];
            if (offset is NAryExpr) {
              var nExpr = (NAryExpr)offset;
              if (nExpr.Fun.FunctionName == "BV32_ADD" &&
                  nExpr.Args[0] is NAryExpr) {
                var n0Expr = (NAryExpr)nExpr.Args[0];
                if (n0Expr.Fun.FunctionName.StartsWith("offset#"))
                  offset = nExpr.Args[1];
              }
            }
            result.Add(offset);
          }

        }

      }

      return result;
    }

    public void AddRaceCheckingDeclarations() {
      foreach (Variable v in NonLocalStateToCheck.getAllNonLocalArrays()) {
        AddRaceCheckingDecsAndProcsForVar(v);
      }
    }

    protected void AddLogAccessProcedure(Variable v, string Access) {
      Procedure LogAccessProcedure = MakeLogAccessProcedureHeader(v, Access);

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      Variable AccessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      Variable AccessOffsetVariable = verifier.MakeOffsetVariable(v.Name, Access);
      Variable AccessValueVariable = GPUVerifier.MakeValueVariable(v.Name, Access, mt.Result);
      Variable AccessSourceVariable = verifier.MakeSourceVariable(v.Name, Access);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable SourceParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_source", mt.Arguments[0]));

      
      Debug.Assert(!(mt.Result is MapType));

      List<Variable> locals = new List<Variable>();
      Variable TrackVariable = new LocalVariable(v.tok, new TypedIdent(v.tok, "track", Microsoft.Boogie.Type.Bool));
      locals.Add(TrackVariable);

      List<BigBlock> bigblocks = new List<BigBlock>();

      List<Cmd> simpleCmds = new List<Cmd>();

      simpleCmds.Add(new HavocCmd(v.tok, new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(v.tok, TrackVariable) })));

      Expr Condition = Expr.And(new IdentifierExpr(v.tok, VariableForThread(1, PredicateParameter)), new IdentifierExpr(v.tok, TrackVariable));

      simpleCmds.Add(MakeConditionalAssignment(VariableForThread(1, AccessHasOccurredVariable),
          Condition, Expr.True));
      simpleCmds.Add(MakeConditionalAssignment(VariableForThread(1, AccessOffsetVariable),
          Condition,
          new IdentifierExpr(v.tok, VariableForThread(1, OffsetParameter))));
      if (!CommandLineOptions.NoBenign) {
        simpleCmds.Add(MakeConditionalAssignment(VariableForThread(1, AccessValueVariable),
          Condition,
          new IdentifierExpr(v.tok, VariableForThread(1, ValueParameter))));
      }
      simpleCmds.Add(MakeConditionalAssignment(VariableForThread(1, AccessSourceVariable),
          Condition,
          new IdentifierExpr(v.tok, VariableForThread(1, SourceParameter))));

      bigblocks.Add(new BigBlock(v.tok, "_LOG_" + Access + "", simpleCmds, null, null));

      Implementation LogAccessImplementation = new Implementation(v.tok, "_LOG_" + Access + "_" + v.Name, new List<TypeVariable>(), LogAccessProcedure.InParams, new List<Variable>(), locals, new StmtList(bigblocks, v.tok));
      GPUVerifier.AddInlineAttribute(LogAccessImplementation);

      LogAccessImplementation.Proc = LogAccessProcedure;

      verifier.Program.TopLevelDeclarations.Add(LogAccessProcedure);
      verifier.Program.TopLevelDeclarations.Add(LogAccessImplementation);
    }


    protected void AddCheckAccessProcedure(Variable v, string Access) {
      Procedure CheckAccessProcedure = MakeCheckAccessProcedureHeader(v, Access);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Debug.Assert(!(mt.Result is MapType));

      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

      if (Access.Equals("READ")) {
        // Check read by thread 2 does not conflict with write by thread 1
        Variable WriteHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "WRITE");
        Variable WriteOffsetVariable = verifier.MakeOffsetVariable(v.Name, "WRITE");
        Expr WriteReadGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
        WriteReadGuard = Expr.And(WriteReadGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteHasOccurredVariable)));
        WriteReadGuard = Expr.And(WriteReadGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteOffsetVariable)),
                                        new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));

        if (!CommandLineOptions.NoBenign) {
          WriteReadGuard = Expr.And(WriteReadGuard, Expr.Neq(
              new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "WRITE", mt.Result))),
              new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))
              ));
        }

        if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
          WriteReadGuard = Expr.And(WriteReadGuard, GPUVerifier.ThreadsInSameGroup());
        }

        WriteReadGuard = Expr.Not(WriteReadGuard);

        Requires NoWriteReadRaceRequires = new Requires(false, WriteReadGuard);
        NoWriteReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "write_read", new List<object>(), null);
        NoWriteReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoWriteReadRaceRequires.Attributes);
        NoWriteReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoWriteReadRaceRequires.Attributes);
        CheckAccessProcedure.Requires.Add(NoWriteReadRaceRequires);

        if (CommandLineOptions.AtomicVsRead) {
          // Check atomic by thread 2 does not conflict with read by thread 1
          Variable AtomicHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "ATOMIC");
          Variable AtomicOffsetVariable = verifier.MakeOffsetVariable(v.Name, "ATOMIC");
          Expr AtomicReadGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
          AtomicReadGuard = Expr.And(AtomicReadGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, AtomicHasOccurredVariable)));
          AtomicReadGuard = Expr.And(AtomicReadGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, AtomicOffsetVariable)),
                                          new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));
          if (!CommandLineOptions.NoBenign) {
            AtomicReadGuard = Expr.And(AtomicReadGuard, Expr.Neq(
                new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "ATOMIC", mt.Result))),
                new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))
                ));
          }

          if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
            AtomicReadGuard = Expr.And(AtomicReadGuard, GPUVerifier.ThreadsInSameGroup());
          }

          AtomicReadGuard = Expr.Not(AtomicReadGuard);

          Requires NoAtomicReadRaceRequires = new Requires(false, AtomicReadGuard);

          NoAtomicReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "atomic_read", new List<object>(), null);
          NoAtomicReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoAtomicReadRaceRequires.Attributes);
          NoAtomicReadRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoAtomicReadRaceRequires.Attributes);
          CheckAccessProcedure.Requires.Add(NoAtomicReadRaceRequires);
        }

      }
      else if (Access.Equals("WRITE")) {

        // Check write by thread 2 does not conflict with write by thread 1
        Variable WriteHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "WRITE");
        Variable WriteOffsetVariable = verifier.MakeOffsetVariable(v.Name, "WRITE");

        Expr WriteWriteGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
        WriteWriteGuard = Expr.And(WriteWriteGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteHasOccurredVariable)));
        WriteWriteGuard = Expr.And(WriteWriteGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteOffsetVariable)),
                                        new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));

        if (!CommandLineOptions.NoBenign) {
          WriteWriteGuard = Expr.And(WriteWriteGuard, Expr.Neq(
              new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "WRITE", mt.Result))),
              new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))));
        }
        
        if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
          WriteWriteGuard = Expr.And(WriteWriteGuard, GPUVerifier.ThreadsInSameGroup());
        }

        WriteWriteGuard = Expr.Not(WriteWriteGuard);
        Requires NoWriteWriteRaceRequires = new Requires(false, WriteWriteGuard);
        NoWriteWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "write_write", new List<object>(), null);
        NoWriteWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoWriteWriteRaceRequires.Attributes);
        NoWriteWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoWriteWriteRaceRequires.Attributes);
        CheckAccessProcedure.Requires.Add(NoWriteWriteRaceRequires);

        // Check write by thread 2 does not conflict with read by thread 1
        Variable ReadHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "READ");
        Variable ReadOffsetVariable = verifier.MakeOffsetVariable(v.Name, "READ");

        Expr ReadWriteGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
        ReadWriteGuard = Expr.And(ReadWriteGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, ReadHasOccurredVariable)));
        ReadWriteGuard = Expr.And(ReadWriteGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, ReadOffsetVariable)),
                                        new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));
        if (!CommandLineOptions.NoBenign) {
          ReadWriteGuard = Expr.And(ReadWriteGuard, Expr.Neq(
              new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "READ", mt.Result))),
              new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))));
        }

        if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
          ReadWriteGuard = Expr.And(ReadWriteGuard, GPUVerifier.ThreadsInSameGroup());
        }

        ReadWriteGuard = Expr.Not(ReadWriteGuard);
        Requires NoReadWriteRaceRequires = new Requires(false, ReadWriteGuard);
        NoReadWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "read_write", new List<object>(), null);
        NoReadWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoReadWriteRaceRequires.Attributes);
        NoReadWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoReadWriteRaceRequires.Attributes);
        CheckAccessProcedure.Requires.Add(NoReadWriteRaceRequires);
        if (CommandLineOptions.AtomicVsWrite) {
          // Check write by thread 2 does not conflict with atomic by thread 1
          Variable AtomicHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "ATOMIC");
          Variable AtomicOffsetVariable = verifier.MakeOffsetVariable(v.Name, "ATOMIC");
          Expr AtomicWriteGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
          AtomicWriteGuard = Expr.And(AtomicWriteGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, AtomicHasOccurredVariable)));
          AtomicWriteGuard = Expr.And(AtomicWriteGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, AtomicOffsetVariable)),
                                          new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));
          if (!CommandLineOptions.NoBenign) {
            AtomicWriteGuard = Expr.And(AtomicWriteGuard, Expr.Neq(
                new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "ATOMIC", mt.Result))),
                new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))
                ));
          }

          if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
            AtomicWriteGuard = Expr.And(AtomicWriteGuard, GPUVerifier.ThreadsInSameGroup());
          }

          AtomicWriteGuard = Expr.Not(AtomicWriteGuard);

          Requires NoAtomicWriteRaceRequires = new Requires(false, AtomicWriteGuard);

          NoAtomicWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "atomic_write", new List<object>(), null);
          NoAtomicWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoAtomicWriteRaceRequires.Attributes);
          NoAtomicWriteRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoAtomicWriteRaceRequires.Attributes);
          CheckAccessProcedure.Requires.Add(NoAtomicWriteRaceRequires);
        }
      }

      else if (Access.Equals("ATOMIC")) {
        if (CommandLineOptions.AtomicVsWrite) {
          // Check atomic by thread 2 does not conflict with write by thread 1
          Variable WriteHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "WRITE");
          Variable WriteOffsetVariable = verifier.MakeOffsetVariable(v.Name, "WRITE");
          Expr WriteAtomicGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
          WriteAtomicGuard = Expr.And(WriteAtomicGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteHasOccurredVariable)));
          WriteAtomicGuard = Expr.And(WriteAtomicGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, WriteOffsetVariable)),
                                          new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));
          if (!CommandLineOptions.NoBenign) {
            WriteAtomicGuard = Expr.And(WriteAtomicGuard, Expr.Neq(
                new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "WRITE", mt.Result))),
                new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))
                ));
          }

          if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
            WriteAtomicGuard = Expr.And(WriteAtomicGuard, GPUVerifier.ThreadsInSameGroup());
          }

          WriteAtomicGuard = Expr.Not(WriteAtomicGuard);

          Requires NoWriteAtomicRaceRequires = new Requires(false, WriteAtomicGuard);

          NoWriteAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "write_atomic", new List<object>(), null);
          NoWriteAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoWriteAtomicRaceRequires.Attributes);
          NoWriteAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoWriteAtomicRaceRequires.Attributes);
          CheckAccessProcedure.Requires.Add(NoWriteAtomicRaceRequires);
        }
        if (CommandLineOptions.AtomicVsRead) {
          // Check atomic by thread 2 does not conflict with read by thread 1
          Variable ReadHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "READ");
          Variable ReadOffsetVariable = verifier.MakeOffsetVariable(v.Name, "READ");
          Expr ReadAtomicGuard = new IdentifierExpr(Token.NoToken, VariableForThread(2, PredicateParameter));
          ReadAtomicGuard = Expr.And(ReadAtomicGuard, new IdentifierExpr(Token.NoToken, VariableForThread(1, ReadHasOccurredVariable)));
          ReadAtomicGuard = Expr.And(ReadAtomicGuard, Expr.Eq(new IdentifierExpr(Token.NoToken, VariableForThread(1, ReadOffsetVariable)),
                                          new IdentifierExpr(Token.NoToken, VariableForThread(2, OffsetParameter))));
          if (!CommandLineOptions.NoBenign) {
            ReadAtomicGuard = Expr.And(ReadAtomicGuard, Expr.Neq(
                new IdentifierExpr(Token.NoToken, VariableForThread(1, GPUVerifier.MakeValueVariable(v.Name, "READ", mt.Result))),
                new IdentifierExpr(Token.NoToken, VariableForThread(2, ValueParameter))
                ));
          }

          if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
            ReadAtomicGuard = Expr.And(ReadAtomicGuard, GPUVerifier.ThreadsInSameGroup());
          }

          ReadAtomicGuard = Expr.Not(ReadAtomicGuard);

          Requires NoReadAtomicRaceRequires = new Requires(false, ReadAtomicGuard);

          NoReadAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "read_atomic", new List<object>(), null);
          NoReadAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "race", new List<object>(), NoReadAtomicRaceRequires.Attributes);
          NoReadAtomicRaceRequires.Attributes = new QKeyValue(Token.NoToken, "array", new List<object>() { v.Name }, NoReadAtomicRaceRequires.Attributes);
          CheckAccessProcedure.Requires.Add(NoReadAtomicRaceRequires);
        }


      }
      verifier.Program.TopLevelDeclarations.Add(CheckAccessProcedure);
    }



    private Variable VariableForThread(int thread, Variable v) {
      return new VariableDualiser(thread, null, null).VisitVariable(v.Clone() as Variable);
    }

    private Expr ExprForThread(int thread, Expr e) {
      return new VariableDualiser(thread, null, null).VisitExpr(e.Clone() as Expr);
    }

    protected void AddLogRaceDeclarations(Variable v, String Access) {
      verifier.FindOrCreateAccessHasOccurredVariable(v.Name, Access);
      verifier.FindOrCreateOffsetVariable(v.Name, Access);
      verifier.FindOrCreateSourceVariable(v.Name, Access);

      if (!CommandLineOptions.NoBenign) {
        Debug.Assert(v.TypedIdent.Type is MapType);
        MapType mt = v.TypedIdent.Type as MapType;
        Debug.Assert(mt.Arguments.Count == 1);
        verifier.FindOrCreateValueVariable(v.Name, Access, mt.Result);
      }
    }


    private static AssignCmd MakeConditionalAssignment(Variable lhs, Expr condition, Expr rhs) {
      List<AssignLhs> lhss = new List<AssignLhs>();
      List<Expr> rhss = new List<Expr>();
      lhss.Add(new SimpleAssignLhs(lhs.tok, new IdentifierExpr(lhs.tok, lhs)));
      rhss.Add(new NAryExpr(rhs.tok, new IfThenElse(rhs.tok), new List<Expr>(new Expr[] { condition, rhs, new IdentifierExpr(lhs.tok, lhs) })));
      return new AssignCmd(lhs.tok, lhss, rhss);
    }

    private Expr MakeAccessedIndex(Variable v, Expr offsetExpr, string Access) {
      Expr result = new IdentifierExpr(v.tok, v.Clone() as Variable);
      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      result = Expr.Select(result,
          new Expr[] { offsetExpr });
      Debug.Assert(!(mt.Result is MapType));
      return result;
    }

    protected void AddRequiresNoPendingAccess(Variable v) {
      IdentifierExpr ReadAccessOccurred1 = new IdentifierExpr(v.tok, new VariableDualiser(1, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "READ")));
      IdentifierExpr WriteAccessOccurred1 = new IdentifierExpr(v.tok, new VariableDualiser(1, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "WRITE")));
      IdentifierExpr AtomicAccessOccurred1 = new IdentifierExpr(v.tok, new VariableDualiser(1, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, "ATOMIC")));

      foreach (var Proc in verifier.KernelProcedures.Keys) {
        Proc.Requires.Add(new Requires(false,Expr.And(Expr.And(Expr.Not(ReadAccessOccurred1), Expr.Not(WriteAccessOccurred1)),Expr.Not(AtomicAccessOccurred1))));
      }
    }

    private void AddRequiresSourceAccessZero(Variable v)
    {
      if (CommandLineOptions.InferSourceLocation) {
        foreach (var Proc in verifier.KernelProcedures.Keys) {
          foreach (string kind in new string[] {"READ","WRITE","ATOMIC"})
            Proc.Requires.Add(new Requires(false, Expr.Eq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateSourceVariable(v.Name, kind)),
                                                                              verifier.Zero())));
        }
      }
    }

    public void AddSourceLocationLoopInvariants(Implementation impl, IRegion region)
    {
      if (!CommandLineOptions.InferSourceLocation) {
        return;
      }

      foreach (string key in AtomicAccessSourceLocations.Keys.Union(WriteAccessSourceLocations.Keys.Union(ReadAccessSourceLocations.Keys)))
      {
        region.AddInvariant(BuildNoAccessInvariant(key, "WRITE"));
        region.AddInvariant(BuildNoAccessInvariant(key, "READ"));
        region.AddInvariant(BuildNoAccessInvariant(key, "ATOMIC"));

        if (WriteAccessSourceLocations.ContainsKey(key))
        {
          region.AddInvariant(BuildPossibleSourceLocationsInvariant(key, "WRITE"));
        }
        else
        {
          region.AddInvariant(BuildAccessOccurredFalseInvariant(key, "WRITE"));
        }

        if (ReadAccessSourceLocations.ContainsKey(key))
        {
          region.AddInvariant(BuildPossibleSourceLocationsInvariant(key, "READ"));
        }
        else
        {
          region.AddInvariant(BuildAccessOccurredFalseInvariant(key, "READ"));
        }

        if (AtomicAccessSourceLocations.ContainsKey(key))
        {
          region.AddInvariant(BuildPossibleSourceLocationsInvariant(key, "ATOMIC"));
        }
        else
        {
          region.AddInvariant(BuildAccessOccurredFalseInvariant(key, "ATOMIC"));
        }
      }
    }

    public void AddStandardSourceVariablePreconditions()
    {
      if (!CommandLineOptions.InferSourceLocation) {
        return;
      }

      foreach (Declaration D in verifier.Program.TopLevelDeclarations.ToList())
      {
        if (!(D is Procedure))
        {
          continue;
        }
        Procedure Proc = D as Procedure;
        if(verifier.ProcedureIsInlined(Proc)) {
          continue;
        }

        foreach (string key in AtomicAccessSourceLocations.Keys.Union(WriteAccessSourceLocations.Keys.Union(ReadAccessSourceLocations.Keys)))
        {
          Proc.Requires.Add(new Requires(false, BuildNoAccessExpr(key, "WRITE")));
          Proc.Requires.Add(new Requires(false, BuildNoAccessExpr(key, "READ")));
          Proc.Requires.Add(new Requires(false, BuildNoAccessExpr(key, "ATOMIC")));

          if (WriteAccessSourceLocations.ContainsKey(key))
          {
            Proc.Requires.Add(new Requires(false, BuildPossibleSourceLocationsExpr(key, "WRITE")));
          }
          else
          {
            Proc.Requires.Add(new Requires(false, BuildAccessOccurredFalseExpr(key, "WRITE")));
          }

          if (ReadAccessSourceLocations.ContainsKey(key))
          {
            Proc.Requires.Add(new Requires(false, BuildPossibleSourceLocationsExpr(key, "READ")));
          }
          else
          {
            Proc.Requires.Add(new Requires(false, BuildAccessOccurredFalseExpr(key, "READ")));
          }

          if (AtomicAccessSourceLocations.ContainsKey(key))
          {
            Proc.Requires.Add(new Requires(false, BuildPossibleSourceLocationsExpr(key, "ATOMIC")));
          }
          else
          {
            Proc.Requires.Add(new Requires(false, BuildAccessOccurredFalseExpr(key, "ATOMIC")));
          }
        }
      }
    }

    public void AddStandardSourceVariablePostconditions()
    {
      if (!CommandLineOptions.InferSourceLocation) {
        return;
      }

      foreach (Declaration D in verifier.Program.TopLevelDeclarations.ToList())
      {
        if (!(D is Procedure)) {
          continue;
        }
        Procedure Proc = D as Procedure;
        if (verifier.ProcedureIsInlined(Proc)) {
          continue;
        }
        foreach (string key in AtomicAccessSourceLocations.Keys.Union(WriteAccessSourceLocations.Keys.Union(ReadAccessSourceLocations.Keys)))
        {
          Proc.Ensures.Add(new Ensures(false, BuildNoAccessExpr(key, "WRITE")));
          Proc.Ensures.Add(new Ensures(false, BuildNoAccessExpr(key, "READ")));
          Proc.Ensures.Add(new Ensures(false, BuildNoAccessExpr(key, "ATOMIC")));

          if (WriteAccessSourceLocations.ContainsKey(key))
          {
            Proc.Ensures.Add(new Ensures(false, BuildPossibleSourceLocationsExpr(key, "WRITE")));
          }
          else
          {
            Proc.Ensures.Add(new Ensures(false, BuildAccessOccurredFalseExpr(key, "WRITE")));
          }

          if (ReadAccessSourceLocations.ContainsKey(key))
          {
            Proc.Ensures.Add(new Ensures(false, BuildPossibleSourceLocationsExpr(key, "READ")));
          }
          else
          {
            Proc.Ensures.Add(new Ensures(false, BuildAccessOccurredFalseExpr(key, "READ")));
          }

          if (AtomicAccessSourceLocations.ContainsKey(key))
          {
            Proc.Ensures.Add(new Ensures(false, BuildPossibleSourceLocationsExpr(key, "ATOMIC")));
          }
          else
          {
            Proc.Ensures.Add(new Ensures(false, BuildAccessOccurredFalseExpr(key, "ATOMIC")));
          }
        }
      }
    }

    private Expr BuildAccessOccurredFalseExpr(string name, string Access)
    {
      return Expr.Imp(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(name, Access)),
                                         Expr.False);
    }
    
    private AssertCmd BuildAccessOccurredFalseInvariant(string name, string Access)
    {
      return new AssertCmd(Token.NoToken, BuildAccessOccurredFalseExpr(name, Access));
    }

    private Expr BuildNoAccessExpr(string name, string Access)
    {
      return Expr.Imp(Expr.Not(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(name, Access))),
                                                   Expr.Eq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateSourceVariable(name, Access)),
                                                           verifier.IntRep.GetLiteral(0, 32)));
    }

    private AssertCmd BuildNoAccessInvariant(string name, string Access)
    {
      return new AssertCmd(Token.NoToken, BuildNoAccessExpr(name, Access));
    }

    private Expr BuildPossibleSourceLocationsExpr(string name, string Access)
    {
      return Expr.Imp(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateAccessHasOccurredVariable(name, Access)),
                                         BuildDisjunctionFromAccessSourceLocations(name, Access));
    }

    private AssertCmd BuildPossibleSourceLocationsInvariant(string name, string Access)
    {
      return new AssertCmd(Token.NoToken, BuildPossibleSourceLocationsExpr(name, Access));
    }

    private Expr BuildDisjunctionFromAccessSourceLocations(string key, string Access)
    {
      List<Expr> sourceLocExprs = new List<Expr>();
      Dictionary<string, List<int>> AccessSourceLocations = (Access.Equals("WRITE")) ? WriteAccessSourceLocations : (Access.Equals("READ") ? ReadAccessSourceLocations : AtomicAccessSourceLocations);
      foreach (int loc in AccessSourceLocations[key])
      {
        sourceLocExprs.Add(Expr.Eq(new IdentifierExpr(Token.NoToken, verifier.FindOrCreateSourceVariable(key, Access)),
                                   verifier.IntRep.GetLiteral(loc, 32)));
      }
      return sourceLocExprs.Aggregate(Expr.Or);
    }

    protected Expr NoAccessExpr(Variable v, string Access, string OneOrTwo) {
      Variable AccessHasOccurred = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      AccessHasOccurred.Name = AccessHasOccurred.Name + "$" + OneOrTwo;
      AccessHasOccurred.TypedIdent.Name = AccessHasOccurred.TypedIdent.Name + "$" + OneOrTwo;
      Expr expr = Expr.Not(new IdentifierExpr(v.tok, AccessHasOccurred));
      return expr;
    }


    protected void AddOffsetsSatisfyPredicatesCandidateInvariant(IRegion region, Variable v, string Access, List<Expr> preds,
            bool SourceLocVersion = false) {
      if (preds.Count != 0) {
        Expr expr = AccessedOffsetsSatisfyPredicatesExpr(v, preds, Access, 1);
        verifier.AddCandidateInvariant(region, expr, "accessed offsets satisfy predicates"
          + (SourceLocVersion ? " (source)" : ""), InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
      }
    }

    private Expr AccessedOffsetsSatisfyPredicatesExpr(Variable v, IEnumerable<Expr> offsets, string Access, int Thread) {
      return Expr.Imp(
              new IdentifierExpr(Token.NoToken, new VariableDualiser(Thread, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access))),
              offsets.Aggregate(Expr.Or));
    }

    private Expr AccessedOffsetIsThreadLocalIdExpr(Variable v, string Access, int Thread) {
      return Expr.Imp(
                new IdentifierExpr(v.tok, new VariableDualiser(Thread, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access))),
                Expr.Eq(new IdentifierExpr(v.tok, new VariableDualiser(Thread, null, null).VisitVariable(verifier.MakeOffsetVariable(v.Name, Access))), 
                  new IdentifierExpr(v.tok, GPUVerifier.MakeThreadId("X", Thread))));
    }

    private Expr GlobalIdExpr(string dimension, int Thread) {
      return new VariableDualiser(Thread, null, null).VisitExpr(verifier.GlobalIdExpr(dimension).Clone() as Expr);
    }

    protected void AddAccessedOffsetInRangeCTimesLocalIdToCTimesLocalIdPlusC(IRegion region, Variable v, Expr constant, string Access) {
      Expr expr = MakeCTimesLocalIdRangeExpression(v, constant, Access, 1);
      verifier.AddCandidateInvariant(region,
          expr, "accessed offset in range [ C*local_id, (C+1)*local_id )", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    private Expr MakeCTimesLocalIdRangeExpression(Variable v, Expr constant, string Access, int Thread) {
      Expr CTimesLocalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
          new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", Thread)));

      Expr CTimesLocalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
          new IdentifierExpr(Token.NoToken, GPUVerifier.MakeThreadId("X", Thread))), constant.Clone() as Expr);

      Expr CTimesLocalIdLeqAccessedOffset = verifier.IntRep.MakeSle(CTimesLocalId, OffsetXExpr(v, Access, Thread));

      Expr AccessedOffsetLtCTimesLocalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, Access, Thread), CTimesLocalIdPlusC);

      return Expr.Imp(
              AccessHasOccurred(v, Access, Thread),
              Expr.And(CTimesLocalIdLeqAccessedOffset, AccessedOffsetLtCTimesLocalIdPlusC));
    }

    private IdentifierExpr AccessHasOccurred(Variable v, string Access, int Thread) {
      return new IdentifierExpr(v.tok, new VariableDualiser(Thread, null, null).VisitVariable(GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access)));
    }

    private IdentifierExpr OffsetXExpr(Variable v, string Access, int Thread) {
      return new IdentifierExpr(v.tok, new VariableDualiser(Thread, null, null).VisitVariable(verifier.MakeOffsetVariable(v.Name, Access)));
    }

    protected void AddAccessedOffsetInRangeCTimesGlobalIdToCTimesGlobalIdPlusC(IRegion region, Variable v, Expr constant, string Access) {
      Expr expr = MakeCTimesGloalIdRangeExpr(v, constant, Access, 1);
      verifier.AddCandidateInvariant(region,
          expr, "accessed offset in range [ C*global_id, (C+1)*global_id )", InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    private Expr MakeCTimesGloalIdRangeExpr(Variable v, Expr constant, string Access, int Thread) {
      Expr CTimesGlobalId = verifier.IntRep.MakeMul(constant.Clone() as Expr,
          GlobalIdExpr("X", Thread));

      Expr CTimesGlobalIdPlusC = verifier.IntRep.MakeAdd(verifier.IntRep.MakeMul(constant.Clone() as Expr,
          GlobalIdExpr("X", Thread)), constant.Clone() as Expr);

      Expr CTimesGlobalIdLeqAccessedOffset = verifier.IntRep.MakeSle(CTimesGlobalId, OffsetXExpr(v, Access, Thread));

      Expr AccessedOffsetLtCTimesGlobalIdPlusC = verifier.IntRep.MakeSlt(OffsetXExpr(v, Access, Thread), CTimesGlobalIdPlusC);

      Expr implication = Expr.Imp(
              AccessHasOccurred(v, Access, Thread),
              Expr.And(CTimesGlobalIdLeqAccessedOffset, AccessedOffsetLtCTimesGlobalIdPlusC));
      return implication;
    }

    private void writeSourceLocToFile(QKeyValue kv, string path) {
      TextWriter tw = new StreamWriter(path, true);
      tw.Write("\n" + QKeyValue.FindIntAttribute(SourceLocationAttributes, "line", -1) 
                    + "#" + QKeyValue.FindIntAttribute(SourceLocationAttributes, "col", -1) 
                    + "#" + QKeyValue.FindStringAttribute(SourceLocationAttributes, "fname") 
                    + "#" + QKeyValue.FindStringAttribute(SourceLocationAttributes, "dir"));
      tw.Close();
    }
    
    protected void AddAccessedOffsetIsThreadLocalIdCandidateRequires(Procedure Proc, Variable v, string Access, int Thread) {
      verifier.AddCandidateRequires(Proc, AccessedOffsetIsThreadLocalIdExpr(v, Access, Thread), InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }

    protected void AddAccessedOffsetIsThreadLocalIdCandidateEnsures(Procedure Proc, Variable v, string Access, int Thread) {
      verifier.AddCandidateEnsures(Proc, AccessedOffsetIsThreadLocalIdExpr(v, Access, Thread), InferenceStages.ACCESS_PATTERN_CANDIDATE_STAGE);
    }



  }



  class FindReferencesToNamedVariableVisitor : StandardVisitor {
    internal bool found = false;
    private string name;

    internal FindReferencesToNamedVariableVisitor(string name) {
      this.name = name;
    }

    public override Variable VisitVariable(Variable node) {
      if (GPUVerifier.StripThreadIdentifier(node.Name).Equals(name)) {
        found = true;
      }
      return base.VisitVariable(node);
    }
  }



}
