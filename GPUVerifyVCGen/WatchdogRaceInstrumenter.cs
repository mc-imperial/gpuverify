//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using Microsoft.Boogie;

    public class WatchdogRaceInstrumenter : RaceInstrumenter
    {
        public WatchdogRaceInstrumenter(GPUVerifier verifier)
            : base(verifier)
        {
        }

        protected override void AddLogAccessProcedure(Variable v, AccessType access)
        {
            // This array should be included in the set of global or group shared arrays that
            // are *not* disabled
            Debug.Assert(Verifier.KernelArrayInfo.ContainsGlobalOrGroupSharedArray(v, false));

            Procedure logAccessProcedure = MakeLogAccessProcedureHeader(v, access);

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);

            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
            Variable accessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, Verifier.IntRep.GetIntType(Verifier.SizeTBits));
            Variable accessValueVariable = RaceInstrumentationUtil.MakeValueVariable(v.Name, access, mt.Result);
            Variable accessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);
            Variable accessAsyncHandleVariable = RaceInstrumentationUtil.MakeAsyncHandleVariable(v.Name, access, Verifier.IntRep.GetIntType(Verifier.SizeTBits));

            Variable predicateParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_P", Type.Bool));
            Variable offsetParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_offset", mt.Arguments[0]));
            Variable valueParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value", mt.Result));
            Variable valueOldParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value_old", mt.Result));
            Variable asyncHandleParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_async_handle", Verifier.IntRep.GetIntType(Verifier.SizeTBits)));

            Debug.Assert(!(mt.Result is MapType));

            Block loggingCommands = new Block(Token.NoToken, "log_access_entry", new List<Cmd>(), new ReturnCmd(Token.NoToken));

            Expr condition = Expr.And(
                new IdentifierExpr(Token.NoToken, MakeTrackingVariable()),
                Expr.Eq(
                    new IdentifierExpr(Token.NoToken, accessOffsetVariable),
                    new IdentifierExpr(Token.NoToken, offsetParameter)));

            if (Verifier.KernelArrayInfo.GetGroupSharedArrays(false).Contains(v))
            {
                condition = Expr.And(Verifier.ThreadsInSameGroup(), condition);
            }

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
            {
                condition = Expr.And(condition, Expr.Eq(new IdentifierExpr(Token.NoToken, accessValueVariable), new IdentifierExpr(Token.NoToken, valueParameter)));
            }

            condition = Expr.And(new IdentifierExpr(Token.NoToken, predicateParameter), condition);

            loggingCommands.Cmds.Add(MakeConditionalAssignment(accessHasOccurredVariable, condition, Expr.True));

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
            {
                loggingCommands.Cmds.Add(MakeConditionalAssignment(
                    accessBenignFlagVariable,
                    condition,
                    Expr.Neq(
                        new IdentifierExpr(Token.NoToken, valueParameter),
                        new IdentifierExpr(Token.NoToken, valueOldParameter))));
            }

            if ((access == AccessType.READ || access == AccessType.WRITE) && Verifier.ArraysAccessedByAsyncWorkGroupCopy[access].Contains(v.Name))
            {
                loggingCommands.Cmds.Add(MakeConditionalAssignment(
                    accessAsyncHandleVariable, condition, Expr.Ident(asyncHandleParameter)));
            }

            Implementation logAccessImplementation =
              new Implementation(
                  Token.NoToken,
                  "_LOG_" + access + "_" + v.Name,
                  new List<TypeVariable>(),
                  logAccessProcedure.InParams,
                  new List<Variable>(),
                  new List<Variable>(),
                  new List<Block> { loggingCommands });
            GPUVerifier.AddInlineAttribute(logAccessImplementation);

            logAccessImplementation.Proc = logAccessProcedure;

            Verifier.Program.AddTopLevelDeclaration(logAccessProcedure);
            Verifier.Program.AddTopLevelDeclaration(logAccessImplementation);
        }

        public override void AddRaceCheckingDeclarations()
        {
            base.AddRaceCheckingDeclarations();
            Verifier.Program.AddTopLevelDeclaration(MakeTrackingVariable());
        }

        private static GlobalVariable MakeTrackingVariable()
        {
            return new GlobalVariable(
                    Token.NoToken, new TypedIdent(Token.NoToken, "_TRACKING", Microsoft.Boogie.Type.Bool));
        }
    }
}
