//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify.InvariantGenerationRules
{
    using Microsoft.Boogie;

    public abstract class InvariantGenerationRule
    {
        protected GPUVerifier Verifier { get; }

        public InvariantGenerationRule(GPUVerifier verifier)
        {
            this.Verifier = verifier;
        }

        public abstract void GenerateCandidates(Implementation impl, IRegion region);
    }
}
