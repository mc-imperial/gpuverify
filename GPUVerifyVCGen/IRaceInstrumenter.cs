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
    using Microsoft.Boogie;

    public interface IRaceInstrumenter
    {
        void AddRaceCheckingCandidateInvariants(Implementation impl, IRegion region);

        void AddKernelPrecondition();

        void AddRaceCheckingInstrumentation();

        void AddRaceCheckingDeclarations();

        BigBlock MakeResetReadWriteSetStatements(Variable v, Expr resetCondition, bool gridBarrier);

        void AddRaceCheckingCandidateRequires(Procedure proc);

        void AddRaceCheckingCandidateEnsures(Procedure proc);

        void AddDefaultLoopInvariants();

        void AddDefaultContracts();
    }
}
