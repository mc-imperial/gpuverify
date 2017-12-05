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
    using System;
    using Microsoft.Boogie;

    public abstract class AccessCollector : StandardVisitor
    {
        public AccessCollector(IKernelArrayInfo state)
        {
            State = state;
        }

        protected IKernelArrayInfo State { get; private set; }

        protected void MultiDimensionalMapError()
        {
            Console.WriteLine("*** Error - multidimensional maps not supported in kernels, use nested maps instead");
            Environment.Exit(1);
        }
    }
}
