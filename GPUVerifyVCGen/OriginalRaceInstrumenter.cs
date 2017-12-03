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

    class OriginalRaceInstrumenter : RaceInstrumenter
    {
        internal OriginalRaceInstrumenter(GPUVerifier verifier)
            : base(verifier)
        {
        }

        protected override void AddLogAccessProcedure(Variable v, AccessType access)
        {
            // This array should be included in the set of global or group shared arrays that
            // are *not* disabled
            Debug.Assert(verifier.KernelArrayInfo.GetGlobalAndGroupSharedArrays(false).Contains(v));

            Procedure logAccessProcedure = MakeLogAccessProcedureHeader(v, access);

            Debug.Assert(v.TypedIdent.Type is MapType);
            MapType mt = v.TypedIdent.Type as MapType;
            Debug.Assert(mt.Arguments.Count == 1);

            Variable accessHasOccurredVariable = GPUVerifier.MakeAccessHasOccurredVariable(v.Name, access);
            Variable accessOffsetVariable = RaceInstrumentationUtil.MakeOffsetVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits));
            Variable accessValueVariable = RaceInstrumentationUtil.MakeValueVariable(v.Name, access, mt.Result);
            Variable accessBenignFlagVariable = GPUVerifier.MakeBenignFlagVariable(v.Name);
            Variable accessAsyncHandleVariable = RaceInstrumentationUtil.MakeAsyncHandleVariable(v.Name, access, verifier.IntRep.GetIntType(verifier.size_t_bits));

            Variable predicateParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_P", Type.Bool));
            Variable offsetParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_offset", mt.Arguments[0]));
            Variable valueParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value", mt.Result));
            Variable valueOldParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_value_old", mt.Result));
            Variable asyncHandleParameter = new LocalVariable(Token.NoToken, new TypedIdent(Token.NoToken, "_async_handle", verifier.IntRep.GetIntType(verifier.size_t_bits)));

            Debug.Assert(!(mt.Result is MapType));

            List<Variable> locals = new List<Variable>();
            Variable trackVariable = new LocalVariable(Token.NoToken,
              new TypedIdent(Token.NoToken, "track", Microsoft.Boogie.Type.Bool));
            locals.Add(trackVariable);

            List<BigBlock> bigblocks = new List<BigBlock>();

            List<Cmd> simpleCmds = new List<Cmd>();

            // Havoc tracking variable
            simpleCmds.Add(new HavocCmd(v.tok, new List<IdentifierExpr>(new IdentifierExpr[] { new IdentifierExpr(v.tok, trackVariable) })));

            Expr condition = Expr.And(new IdentifierExpr(v.tok, predicateParameter),
              new IdentifierExpr(v.tok, trackVariable));

            if (verifier.KernelArrayInfo.GetGroupSharedArrays(false).Contains(v))
            {
                condition = Expr.And(GPUVerifier.ThreadsInSameGroup(), condition);
            }

            simpleCmds.Add(MakeConditionalAssignment(accessHasOccurredVariable,
                condition, Expr.True));
            simpleCmds.Add(MakeConditionalAssignment(accessOffsetVariable,
                condition,
                new IdentifierExpr(v.tok, offsetParameter)));
            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access.IsReadOrWrite())
            {
                simpleCmds.Add(MakeConditionalAssignment(accessValueVariable,
                  condition,
                  new IdentifierExpr(v.tok, valueParameter)));
            }

            if (!GPUVerifyVCGenCommandLineOptions.NoBenign && access == AccessType.WRITE)
            {
                simpleCmds.Add(MakeConditionalAssignment(accessBenignFlagVariable,
                  condition,
                  Expr.Neq(new IdentifierExpr(v.tok, valueParameter),
                    new IdentifierExpr(v.tok, valueOldParameter))));
            }

            if ((access == AccessType.READ || access == AccessType.WRITE) &&
              verifier.ArraysAccessedByAsyncWorkGroupCopy[access].Contains(v.Name))
            {
                simpleCmds.Add(MakeConditionalAssignment(accessAsyncHandleVariable,
                  condition,
                  Expr.Ident(asyncHandleParameter)));
            }

            bigblocks.Add(new BigBlock(v.tok, "_LOG_" + access + "", simpleCmds, null, null));

            Implementation logAccessImplementation = new Implementation(v.tok, "_LOG_" + access + "_" + v.Name, new List<TypeVariable>(), logAccessProcedure.InParams, new List<Variable>(), locals, new StmtList(bigblocks, v.tok));
            GPUVerifier.AddInlineAttribute(logAccessImplementation);

            logAccessImplementation.Proc = logAccessProcedure;

            verifier.Program.AddTopLevelDeclaration(logAccessProcedure);
            verifier.Program.AddTopLevelDeclaration(logAccessImplementation);
        }
    }
}
