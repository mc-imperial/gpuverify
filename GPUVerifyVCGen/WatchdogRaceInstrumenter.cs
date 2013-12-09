using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class WatchdogRaceInstrumenter : RaceInstrumenter
  {
    internal WatchdogRaceInstrumenter(GPUVerifier verifier) : base(verifier) {

    }

    protected override void AddLogAccessProcedure(Variable v, AccessType Access) {
      Procedure LogAccessProcedure = MakeLogAccessProcedureHeader(v, Access);

      Debug.Assert(v.TypedIdent.Type is MapType);
      MapType mt = v.TypedIdent.Type as MapType;
      Debug.Assert(mt.Arguments.Count == 1);

      Variable AccessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, Access);
      Variable AccessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(32));
      Variable AccessValueVariable = RaceInstrumentationUtil.MakeValueVariable(v.Name, Access, mt.Result);
      Variable AccessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable ValueOldParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value_old", mt.Result));

      Debug.Assert(!(mt.Result is MapType));

      Block LoggingCommands = new Block(Token.NoToken, "log_access_entry", new List<Cmd>(), new ReturnCmd(Token.NoToken));

      Expr Condition = Expr.And(new IdentifierExpr(Token.NoToken, MakeTrackingVariable()), Expr.Eq(new IdentifierExpr(Token.NoToken, AccessOffsetVariable),
                                         new IdentifierExpr(Token.NoToken, OffsetParameter)));
      if(!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        Condition = Expr.And(Condition, Expr.Eq(new IdentifierExpr(Token.NoToken, AccessValueVariable), new IdentifierExpr(Token.NoToken, ValueParameter)));
      }

      Condition = Expr.And(new IdentifierExpr(Token.NoToken, PredicateParameter), Condition);

      LoggingCommands.Cmds.Add(MakeConditionalAssignment(AccessHasOccurredVariable, Condition, Expr.True));
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        LoggingCommands.Cmds.Add(MakeConditionalAssignment(AccessBenignFlagVariable,
          Condition,
          Expr.Neq(new IdentifierExpr(Token.NoToken, ValueParameter),
            new IdentifierExpr(Token.NoToken, ValueOldParameter))));
      }

      Implementation LogAccessImplementation = 
        new Implementation(Token.NoToken, "_LOG_" + Access + "_" + v.Name,
          new List<TypeVariable>(),
          LogAccessProcedure.InParams, new List<Variable>(), new List<Variable>(),
          new List<Block> { LoggingCommands } );
      GPUVerifier.AddInlineAttribute(LogAccessImplementation);

      LogAccessImplementation.Proc = LogAccessProcedure;

      verifier.Program.TopLevelDeclarations.Add(LogAccessProcedure);
      verifier.Program.TopLevelDeclarations.Add(LogAccessImplementation);
    }

    public override void AddRaceCheckingDeclarations() {
      base.AddRaceCheckingDeclarations();
      verifier.Program.TopLevelDeclarations.Add(MakeTrackingVariable());
    }

    private static GlobalVariable MakeTrackingVariable()
    {
      return new GlobalVariable(
              Token.NoToken, new TypedIdent(Token.NoToken, "_TRACKING", Microsoft.Boogie.Type.Bool));
    }

  }
}
