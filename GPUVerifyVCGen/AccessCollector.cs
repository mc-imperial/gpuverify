//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.Boogie;

namespace GPUVerify
{
    abstract class AccessCollector : StandardVisitor
    {
        protected IKernelArrayInfo State;

        public AccessCollector(IKernelArrayInfo State)
        {
            this.State = State;
        }

        protected void MultiDimensionalMapError()
        {
            Console.WriteLine("*** Error - multidimensional maps not supported in kernels, use nested maps instead");
            Environment.Exit(1);
        }

    }
}
