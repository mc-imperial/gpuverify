using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  class WatchdogSingleRaceInstrumenter : RaceInstrumenter
  {
    internal WatchdogSingleRaceInstrumenter(GPUVerifier verifier) : base(verifier) {

    }

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

      // TODO: adapt to include "tracking races" variable, reset at barriers

      Block b = new Block(Token.NoToken, "log_access_entry", new List<Cmd>(), new ReturnCmd(Token.NoToken));

      Expr Condition = Expr.And(new IdentifierExpr(Token.NoToken, PredicateParameter), Expr.Eq(new IdentifierExpr(Token.NoToken, AccessOffsetVariable),
                                         new IdentifierExpr(Token.NoToken, OffsetParameter)));

      b.Cmds.Add(MakeConditionalAssignment(AccessHasOccurredVariable, Condition, Expr.True));
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access.isReadOrWrite()) {
        b.Cmds.Add(MakeConditionalAssignment(AccessValueVariable,
          Condition,
          new IdentifierExpr(v.tok, ValueParameter)));
      }
      if (!GPUVerifyVCGenCommandLineOptions.NoBenign && Access == AccessType.WRITE) {
        b.Cmds.Add(MakeConditionalAssignment(AccessBenignFlagVariable,
          Condition,
          Expr.Neq(new IdentifierExpr(v.tok, ValueParameter),
            new IdentifierExpr(v.tok, ValueOldParameter))));
      }

      Implementation LogAccessImplementation = 
        new Implementation(Token.NoToken, "_LOG_" + Access + "_" + v.Name,
          new List<TypeVariable>(),
          LogAccessProcedure.InParams, new List<Variable>(), new List<Variable>(),
          new List<Block> { b} );
      GPUVerifier.AddInlineAttribute(LogAccessImplementation);

      LogAccessImplementation.Proc = LogAccessProcedure;

      verifier.Program.TopLevelDeclarations.Add(LogAccessProcedure);
      verifier.Program.TopLevelDeclarations.Add(LogAccessImplementation);
    }

  }
}
