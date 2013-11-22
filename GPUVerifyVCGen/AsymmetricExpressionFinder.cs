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
    class AsymmetricExpressionFinder : StandardVisitor
    {
        private bool found = false;

        internal bool foundAsymmetricExpr()
        {
            return found;
        }

        static HashSet<string> AsymmetricNamePrefixes = new HashSet<string> {
          "_READ_HAS_OCCURRED", "_READ_OFFSET", "_READ_VALUE",
          "_WRITE_HAS_OCCURRED", "_WRITE_OFFSET", "_WRITE_VALUE",
          "_ATOMIC_HAS_OCCURRED", "_ATOMIC_OFFSET",
          "_WRITE_READ_BENIGN_FLAG"
        };

        public override Variable VisitVariable(Variable node)
        {
          foreach(var prefix in AsymmetricNamePrefixes) {
            if(node.TypedIdent.Name.StartsWith(prefix)) {
                found = true;
            }
          }
          return node;
        }

    }
}
