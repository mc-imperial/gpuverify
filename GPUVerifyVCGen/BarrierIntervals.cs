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
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify
{

  internal enum BarrierStrength {
    GROUP_SHARED, GLOBAL, ALL
  }

  internal class BarrierIntervalsAnalysis
  {

    private GPUVerifier verifier;
    private Dictionary<Implementation, HashSet<BarrierInterval>> intervals;
    private BarrierStrength strength;

    internal BarrierIntervalsAnalysis(GPUVerifier verifier, BarrierStrength strength) {
      this.verifier = verifier;
      this.strength = strength;
    }

    internal void Compute() {
      Debug.Assert(intervals == null);
      intervals = new Dictionary<Implementation,HashSet<BarrierInterval>>();
      foreach(var impl in verifier.KernelProcedures.Values) {
        intervals[impl] = ComputeBarrierIntervals(impl);
      }

      /*foreach(var impl in KernelProcedures.Values) {
        IntraProceduralLiveVariableAnalysis iplva = new IntraProceduralLiveVariableAnalysis(Program, impl);
        iplva.RunAnalysis();
      }*/
    }

    private HashSet<BarrierInterval> ComputeBarrierIntervals(Implementation impl)
    {
      HashSet<BarrierInterval> result = new HashSet<BarrierInterval>();

      ExtractCommandsIntoBlocks(impl, Item => (Item is CallCmd && GPUVerifier.IsBarrier(((CallCmd)Item).Proc)));
      Graph<Block> cfg = Program.GraphFromImpl(impl);
      Graph<Block> dual = cfg.Dual(new Block());
      DomRelation<Block> dom = cfg.DominatorMap;
      DomRelation<Block> pdom = dual.DominatorMap;

      foreach (var dominator in impl.Blocks.Where(Item => StartsWithUnconditionalBarrier(Item)))
      {
        Block smallestBarrierIntervalEnd = null;
        foreach (var postdominator in impl.Blocks.Where(Item => Item != dominator &&
            StartsWithUnconditionalBarrier(Item) &&
            dom.DominatedBy(Item, dominator) &&
            pdom.DominatedBy(dominator, Item)))
        {
          if (smallestBarrierIntervalEnd == null || dom.DominatedBy(smallestBarrierIntervalEnd, postdominator))
          {
            smallestBarrierIntervalEnd = postdominator;
          }
          else
          {
            Debug.Assert(dom.DominatedBy(postdominator, smallestBarrierIntervalEnd));
          }
        }
        if (smallestBarrierIntervalEnd != null)
        {
          result.Add(new BarrierInterval(dominator, smallestBarrierIntervalEnd, dom, pdom, impl));
        }
      }

      return result;
    }

    private bool StartsWithUnconditionalBarrier(Block b)
    {
      if(b.Cmds.Count == 0) {
        return false;
      }
      CallCmd c = b.Cmds[0] as CallCmd;
      if(c == null || !GPUVerifier.IsBarrier(c.Proc)) {
        return false;
      }

      var BarrierProcedure = c.Proc;

      if(!verifier.uniformityAnalyser.IsUniform(BarrierProcedure.Name)) {
        // We may be able to do better in this case, but for now we conservatively say no
        return false;  
      }

      if(BarrierHasNonUniformArgument(BarrierProcedure)) {
        // Also we may be able to do better in this case, but for now we conservatively say no
        return false;
      }
      
      Debug.Assert(c.Ins.Count() == 2);
      if(strength == BarrierStrength.GROUP_SHARED || strength == BarrierStrength.ALL) {
        if(!c.Ins[0].Equals(verifier.IntRep.GetLiteral(1, 1))) {
          return false;
        }
      } else if(strength == BarrierStrength.GLOBAL || strength == BarrierStrength.ALL) {
        if(!c.Ins[1].Equals(verifier.IntRep.GetLiteral(1, 1))) {
          return false;
        }
      } else {
        // All cases should be covered by the above
        Debug.Assert(false);
      }
      return true;
    }

    private bool BarrierHasNonUniformArgument(Procedure BarrierProcedure)
    {
      foreach(var v in BarrierProcedure.InParams) {
        if(!verifier.uniformityAnalyser.IsUniform(BarrierProcedure.Name, GVUtil.StripThreadIdentifier(v.Name))) {
          return true;
        }
      }
      return false;
    }

    void ExtractCommandsIntoBlocks(Implementation impl, Func<Cmd, bool> Predicate) {
      Dictionary<Block, Block> oldToNew = new Dictionary<Block,Block>();
      HashSet<Block> newBlocks = new HashSet<Block>();
      HashSet<Block> removedBlocks = new HashSet<Block>();
      Block newEntryBlock = null;

      foreach (Block b in impl.Blocks)
      {
        List<List<Cmd>> partition = InterproceduralReachabilityGraph.PartitionCmdsAccordingToPredicate(b.Cmds, Predicate);
        if(partition.Count == 1) {
          // Nothing to do: either no command in this block matches the predicate, or there
          // is only one command in the block
          continue;
        }

        removedBlocks.Add(b);

        List<Block> newBlocksForPartitionEntry = new List<Block>();
        for(int i = 0; i < partition.Count; i++) {
          newBlocksForPartitionEntry.Add(new Block(b.tok, "__partitioned_block_" + b.Label + "_" + i, partition[i], null));
          newBlocks.Add(newBlocksForPartitionEntry[i]);
          if(i > 0) {
            newBlocksForPartitionEntry[i - 1].TransferCmd = new GotoCmd(b.tok, new List<string> { newBlocksForPartitionEntry[i].Label }, new List<Block> { newBlocksForPartitionEntry[i] });
          }
          if(i == partition.Count - 1) {
            newBlocksForPartitionEntry[i].TransferCmd = b.TransferCmd;
          }
        }
        oldToNew[b] = newBlocksForPartitionEntry[0];
        if(b == impl.Blocks[0]) {
          Debug.Assert(newEntryBlock == null);
          newEntryBlock = newBlocksForPartitionEntry[0];
        }
      }

      impl.Blocks.RemoveAll(Item => removedBlocks.Contains(Item));

      if(newEntryBlock != null) {
        // Replace the entry block if necessary
        impl.Blocks.Insert(0, newEntryBlock);
        newBlocks.Remove(newEntryBlock);
      }

      // Add all new block that do not replace the entry block
      impl.Blocks.AddRange(newBlocks);

      foreach (var gc in impl.Blocks.Select(Item => Item.TransferCmd).OfType<GotoCmd>()) {
        Debug.Assert(gc.labelNames.Count == gc.labelTargets.Count);
        for(int i = 0; i < gc.labelTargets.Count; i++) {
          if(oldToNew.ContainsKey(gc.labelTargets[i])) {
            Block newBlock = oldToNew[gc.labelTargets[i]];
            gc.labelTargets[i] = newBlock;
            gc.labelNames[i] = newBlock.Label;
          }
          if(!impl.Blocks.Contains(gc.labelTargets[i])) {
            Console.WriteLine("Block " + gc.labelTargets[i] + " still around!");
          }
          Debug.Assert(impl.Blocks.Contains(gc.labelTargets[i]));
        }
      }

    }


    internal void RemoveRedundantReads()
    {
      if(strength == BarrierStrength.GLOBAL) {
        return;
      }

      foreach(BarrierInterval interval in intervals.Values.SelectMany(Item => Item)) {
        var WrittenGroupSharedArrays = interval.FindWrittenGroupSharedArrays(verifier);
        RemoveReads(interval.Blocks,
          verifier.KernelArrayInfo.getGroupSharedArrays().Where(Item =>
             !WrittenGroupSharedArrays.Contains(Item)));
      }
    }

    internal void RemoveReads(IEnumerable<Block> blocks, IEnumerable<Variable> arrays) {
      foreach(var b in blocks) {
        List<Cmd> newCmds = new List<Cmd>();
        foreach(var c in b.Cmds) {
          CallCmd callCmd = c as CallCmd;
          if(callCmd != null) {
            Variable v;
            verifier.TryGetArrayFromLogProcedure(callCmd.callee, AccessType.READ, out v);
            if(v == null) {
              verifier.TryGetArrayFromCheckProcedure(callCmd.callee, AccessType.READ, out v);
            }
            if(v != null && arrays.Contains(v)) {
              continue;
            }
          }
          newCmds.Add(c);
        }
        b.Cmds = newCmds;
      }
    }

  }


  internal class BarrierInterval {
    private Block start;
    private Block end;
    private IEnumerable<Block> blocks;

    public IEnumerable<Block> Blocks {
      get { return blocks; }
    }

    public BarrierInterval(Block start, Block end, DomRelation<Block> dom, DomRelation<Block> pdom, Implementation impl)
    {
      this.start = start;
      this.end = end;
      blocks = impl.Blocks.Where(Item =>
        Item != end &&
        dom.DominatedBy(Item, start) &&
        pdom.DominatedBy(Item, end)).ToList();
    }

    internal HashSet<Variable> FindWrittenGroupSharedArrays(GPUVerifier verifier)
    {
      HashSet<Variable> result = new HashSet<Variable>();
      foreach (var m in Blocks.Select(Item => Item.Cmds).SelectMany(Item => Item).OfType<CallCmd>()
        .Select(Item => Item.Proc.Modifies).SelectMany(Item => Item))
      {
        // m is a variable modified by a call in the barrier interval
        Variable v;
        if(verifier.TryGetArrayFromAccessHasOccurred(GVUtil.StripThreadIdentifier(m.Name), AccessType.WRITE, out v)) {
          if (verifier.KernelArrayInfo.getGroupSharedArrays().Contains(v)) {
            result.Add(v);
          }
        }
      }
      return result;
    }

  }

}
