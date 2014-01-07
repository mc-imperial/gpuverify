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
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify
{
    public class ControlDependence
    {
        private Dictionary<Block, Block> ImmediatePostdominators = new Dictionary<Block, Block>();
        private Dictionary<Block, HashSet<Block>> Dependences = new Dictionary<Block, HashSet<Block>>();

        public ControlDependence (Implementation impl)
        {
            Graph<Block> cfg = Program.GraphFromImpl(impl);
            Graph<Block> reverse = cfg.Dual(new Block());
            // Assume an empty control-dependence relation
            foreach (Block block in cfg.Nodes)
            {
                Dependences[block] = new HashSet<Block>();
            }
            // Build up an immediate post-dominator map to ease control dependence calculation
            ComputeImmediatePostdominators(reverse, reverse.DominatorMap);
            ComputeControlDependences(reverse, reverse.DominatorMap);
        }

        public HashSet<Block> GetControllingNodes ()
        {
            HashSet<Block> nodes = new HashSet<Block>();
            foreach (KeyValuePair<Block, HashSet<Block>> it in Dependences)
            {
                if (it.Value.Count > 0)
                    nodes.Add(it.Key);
            }
            return nodes;
        }

        private void ComputeImmediatePostdominators (Graph<Block> cfg, DomRelation<Block> postdom)
        {
            foreach (Block block in cfg.Nodes)
            {
                if (postdom.ImmediateDominatorMap.ContainsKey(block))
                {
                    foreach (Block dominated in postdom.ImmediateDominatorMap[block])
                    {
                        ImmediatePostdominators[dominated] = block;
                    }
                }
            }
        }

        private void ComputeControlDependences (Graph<Block> cfg, DomRelation<Block> postdom)
        {
            // We use the property that, if X is in the post-dominance frontier of Y, Y is control
            // dependent on X. The following computes post-dominance frontier information
            foreach (Block block in cfg.Nodes)
            {
                IEnumerable<Block> predecessors = cfg.Predecessors(block);
                if (predecessors.Count() > 1)
                {
                    Block immediatePostdom = ImmediatePostdominators[block];
                    foreach (Block pred in predecessors)
                    {
                        Block runner = pred;
                        while (runner != immediatePostdom)
                        {
                            Dependences[block].Add(runner);
                            runner = ImmediatePostdominators[runner];
                        }
                    }
                }
            }
        }
    }
}

