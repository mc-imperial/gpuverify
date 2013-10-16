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

namespace GPUVerify
{
  public sealed class AccessType {

    private readonly String name;

    public static readonly AccessType READ = new AccessType ("READ");
    public static readonly AccessType WRITE = new AccessType ("WRITE");
    public static readonly AccessType ATOMIC = new AccessType ("ATOMIC");

    public static readonly IEnumerable<AccessType> Types = new List<AccessType> { READ, WRITE, ATOMIC };

    private AccessType(String name) {
      this.name = name;
    }

    public override String ToString() {
      return name;
    }

    public bool isReadOrWrite() {
      return this == READ || this == WRITE;
    }
  }
}
