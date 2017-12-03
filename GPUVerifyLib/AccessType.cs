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
    using System.Collections.Generic;
    using System.Diagnostics;

    public sealed class AccessType
    {
        public static readonly AccessType READ = new AccessType("READ");
        public static readonly AccessType WRITE = new AccessType("WRITE");
        public static readonly AccessType ATOMIC = new AccessType("ATOMIC");

        public static readonly IEnumerable<AccessType> Types = new List<AccessType> { READ, WRITE, ATOMIC };

        private readonly string name;

        private AccessType(string name)
        {
            this.name = name;
        }

        public static AccessType Create(string access)
        {
            if (access.ToUpper() == "READ")
                return READ;

            if (access.ToUpper() == "WRITE")
                return WRITE;

            if (access.ToUpper() == "ATOMIC")
                return ATOMIC;

            throw new NotSupportedException("Unknown access type: " + access);
        }

        public override string ToString()
        {
            return name;
        }

        public bool IsReadOrWrite()
        {
            return this == READ || this == WRITE;
        }

        public string Direction()
        {
            if (this == READ)
                return "from";

            if (this == WRITE)
                return "to";

            Debug.Assert(this == ATOMIC);
            return "on";
        }
    }
}
