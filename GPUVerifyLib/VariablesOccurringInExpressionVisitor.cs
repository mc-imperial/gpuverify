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
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
    public class VariablesOccurringInExpressionVisitor : StandardVisitor
    {
        private HashSet<Variable> variables = new HashSet<Variable>();

        public IEnumerable<Microsoft.Boogie.Variable> GetVariables()
        {
            return variables;
        }

        public override Variable VisitVariable(Variable node)
        {
            variables.Add(node);
            return base.VisitVariable(node);
        }

    }
}
