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
    using System.Diagnostics;
    using Microsoft.Boogie;

    public interface IRegion
    {
        object Identifier();

        IEnumerable<Cmd> Cmds();

        IEnumerable<object> CmdsChildRegions();

        IEnumerable<IRegion> SubRegions();

        IEnumerable<Block> PreHeaders();

        Block Header();

        IEnumerable<Block> SubBlocks();

        Expr Guard();

        void AddInvariant(PredicateCmd pc);

        void AddLoopInvariantDisabledTag();

        List<PredicateCmd> RemoveInvariants();

        HashSet<Variable> PartitionVariablesOfHeader();

        HashSet<Variable> PartitionVariablesOfRegion();
    }

    public static class RegionExtensions
    {
        public static HashSet<Variable> GetModifiedVariables(this IRegion region)
        {
            HashSet<Variable> result = new HashSet<Variable>();

            foreach (Cmd c in region.Cmds())
            {
                List<Variable> vars = new List<Variable>();
                c.AddAssignedVariables(vars);
                foreach (Variable v in vars)
                {
                    Debug.Assert(v != null);
                    result.Add(v);
                }
            }

            return result;
        }
    }
}
