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
      Variable AccessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, Access, verifier.IntRep.GetIntType(verifier.size_t_bits));
      Variable AccessValueVariable = RaceInstrumentationUtil.MakeValueVariable(v.Name, Access, mt.Result);
      Variable AccessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);

      Variable PredicateParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_P", Microsoft.Boogie.Type.Bool));
      Variable OffsetParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_offset", mt.Arguments[0]));
      Variable ValueParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value", mt.Result));
      Variable ValueOldParameter = new LocalVariable(v.tok, new TypedIdent(v.tok, "_value_old", mt.Result));

      Debug.Assert(!(mt.Result is MapType));

      List<Variable> locals = new List<Variable>();
      string TrackName = (GPUVerifyVCGenCommandLineOptions.InvertedTracking ?
                              "do_not_track" : "track");
      Variable TrackVariable = new LocalVariable(v.tok, 
        new TypedIdent(v.tok, TrackName, Microsoft.Boogie.Type.Bool));
      locals.Add(TrackVariable);

      List<BigBlock> bigblocks = new List<BigBlock>();

      List<Cmd> simpleCmds = new List<Cmd>();

      if (GPUVerifyVCGenCommandLineOptions.OnlyWarp) {
        // track := true
        // Or equivalently: do_not_track := false
        Expr rhs = (GPUVerifyVCGenCommandLineOptions.InvertedTracking ? 
                    Expr.False : Expr.True);
        simpleCmds.Add(AssignCmd.SimpleAssign(v.tok, Expr.Ident(TrackVariable), rhs));
      }
      else {
        // Havoc tracking variable
        simpleCmds.Add(new HavocCmd(v.tok, new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(v.tok, TrackVariable) })));
      }

      Expr Condition = Expr.And(new IdentifierExpr(v.tok, PredicateParameter),
        (GPUVerifyVCGenCommandLineOptions.InvertedTracking ? 
         Expr.Not(new IdentifierExpr(v.tok, TrackVariable)) : 
                  new IdentifierExpr(v.tok, TrackVariable)));

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

  }
}
