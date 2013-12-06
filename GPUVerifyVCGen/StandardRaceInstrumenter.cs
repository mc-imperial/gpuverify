using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class StandardRaceInstrumenter : RaceInstrumenter
  {

    internal StandardRaceInstrumenter(GPUVerifier verifier) : base(verifier) { }

    protected override void AddLogAccessProcedure(Variable v, AccessType Access) {
      Procedure LogAccessProcedure = MakeLogAccessProcedureHeader(v, Access);

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      Variable AccessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      Variable AccessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(32));
      Variable AccessValueVariable = GPUVerifier.MakeValueVariable(v.Name, Access, mt.Result);
      Variable AccessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable ValueOldParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value_old", mt.Result));

      Debug.Assert(!(mt.Result is MapType));

      List<Variable> locals = new List<Variable>();
      Variable TrackVariable = new LocalVariable(v.tok, new TypedIdent(v.tok, "track", Microsoft.Boogie.Type.Bool));
      locals.Add(TrackVariable);

      List<BigBlock> bigblocks = new List<BigBlock>();

      List<Cmd> simpleCmds = new List<Cmd>();

      simpleCmds.Add(new HavocCmd(v.tok, new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(v.tok, TrackVariable) })));

      Expr Condition = Expr.And(new IdentifierExpr(v.tok, PredicateParameter), new IdentifierExpr(v.tok, TrackVariable));

      simpleCmds.Add(MakeConditionalAssignment(AccessHasOccurredVariable,
          Condition, Expr.True));
      simpleCmds.Add(MakeConditionalAssignment(AccessOffsetVariable,
          Condition,
          new IdentifierExpr(v.tok, OffsetParameter)));
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        simpleCmds.Add(MakeConditionalAssignment(AccessValueVariable,
          Condition,
          new IdentifierExpr(v.tok, ValueParameter)));
      }
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        simpleCmds.Add(MakeConditionalAssignment(AccessBenignFlagVariable,
          Condition,
          Expr.Neq(new IdentifierExpr(v.tok, ValueParameter),
            new IdentifierExpr(v.tok, ValueOldParameter))));
      }

      bigblocks.Add(new BigBlock(v.tok, "_LOG_" + Access + "", simpleCmds, null, null));

      Implementation LogAccessImplementation = new Implementation(v.tok, "_LOG_" + Access + "_" + v.Name, new List<TypeVariable>(), LogAccessProcedure.InParams, new List<Variable>(), locals, new StmtList(bigblocks, v.tok));
      GPUVerifier.AddInlineAttribute(LogAccessImplementation);

      LogAccessImplementation.Proc = LogAccessProcedure;

      verifier.Program.TopLevelDeclarations.Add(LogAccessProcedure);
      verifier.Program.TopLevelDeclarations.Add(LogAccessImplementation);
    }

    protected override void AddCheckAccessProcedure(Variable v, AccessType Access) {
      Procedure CheckAccessProcedure = MakeCheckAccessProcedureHeader(v, Access);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);
      Debug.Assert(!(mt.Result is MapType));

      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));

      if (Access == AccessType.READ) {
        Variable WriteReadBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

        Expr NoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          NoBenignTest = new IdentifierExpr(Token.NoToken, WriteReadBenignFlagVariable);
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, NoBenignTest, AccessType.WRITE, "write_read");

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.ATOMIC, "atomic_read");
        }
      }
      else if (Access == AccessType.WRITE) {
        Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));

        Expr WriteNoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          WriteNoBenignTest = Expr.Neq(
              new IdentifierExpr(Token.NoToken, GPUVerifier.MakeValueVariable(v.Name, AccessType.WRITE, mt.Result)),
              new IdentifierExpr(Token.NoToken, ValueParameter));
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, WriteNoBenignTest, AccessType.WRITE, "write_write");

        Expr ReadNoBenignTest = null;

        if (!GPUVerifyVCGenCommandLineOptions.NoBenign) {
          ReadNoBenignTest = Expr.Neq(
              new IdentifierExpr(Token.NoToken, GPUVerifier.MakeValueVariable(v.Name, AccessType.READ, mt.Result)),
              new IdentifierExpr(Token.NoToken, ValueParameter));
        }

        AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, ReadNoBenignTest, AccessType.READ, "read_write");

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.ATOMIC, "atomic_write");
        }
      }
      else if (Access == AccessType.ATOMIC) {
        if (GPUVerifyVCGenCommandLineOptions.AtomicVsWrite) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.WRITE, "write_atomic");
        }

        if (GPUVerifyVCGenCommandLineOptions.AtomicVsRead) {
          AddCheckAccessCheck(v, CheckAccessProcedure, PredicateParameter, OffsetParameter, null, AccessType.READ, "read_atomic");
        }
      }

      verifier.Program.TopLevelDeclarations.Add(CheckAccessProcedure);
    }


  }
}
