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
    using Microsoft.Boogie;

    public class NullRaceInstrumenter : IRaceInstrumenter
    {
        public void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region)
        {
        }

        public void AddKernelPrecondition()
        {
        }

        public void AddRaceCheckingInstrumentation()
        {
        }

        public Microsoft.Boogie.BigBlock MakeResetReadWriteSetStatements(Variable v, Expr resetCondition)
        {
            return new BigBlock(Token.NoToken, null, new List<Cmd>(), null, null);
        }

        public void AddRaceCheckingCandidateRequires(Procedure proc)
        {
        }

        public void AddRaceCheckingCandidateEnsures(Procedure proc)
        {
        }

        public void AddRaceCheckingDeclarations()
        {
        }

        public void AddSourceLocationLoopInvariants(Implementation impl, IRegion region)
        {
        }

        public void AddStandardSourceVariablePreconditions()
        {
        }

        public void AddStandardSourceVariablePostconditions()
        {
        }

        public void AddDefaultLoopInvariants()
        {
        }

        public void AddDefaultContracts()
        {
        }
    }
}
