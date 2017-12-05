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
    public interface IConstantWriteInstrumenter
    {
        void AddConstantWriteInstrumentation();
    }
}
