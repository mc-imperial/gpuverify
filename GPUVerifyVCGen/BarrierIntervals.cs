using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify
{
  internal class BarrierIntervalsAnalysis
  {

    private GPUVerifier verifier;

    internal BarrierIntervalsAnalysis(GPUVerifier verifier) {
      this.verifier = verifier;
    }


    internal void Compute() {
      Dictionary<Implementation, HashSet<BarrierInterval>> result = new Dictionary<Implementation,HashSet<BarrierInterval>>();
      foreach(var impl in verifier.KernelProcedures.Values) {
        ComputeBarrierIntervals(impl);
      }

      /*foreach(var impl in KernelProcedures.Values) {
        IntraProceduralLiveVariableAnalysis iplva = new IntraProceduralLiveVariableAnalysis(Program, impl);
        iplva.RunAnalysis();
      }*/
    }

    private void ComputeBarrierIntervals(Implementation impl)
    {
      HashSet<BarrierInterval> result = new HashSet<BarrierInterval>();

      ExtractCommandsIntoBlocks(impl, Item => (Item is CallCmd && ((CallCmd)Item).Proc == verifier.BarrierProcedure));
      Graph<Block> cfg = Program.GraphFromImpl(impl);
      Graph<Block> dual = cfg.Dual(new Block());
      DomRelation<Block> dom = cfg.DominatorMap;
      DomRelation<Block> pdom = dual.DominatorMap;

      foreach (var dominator in impl.Blocks.Where(Item => StartsWithUnconditionalBarrier(Item)))
      {
        Block smallestBarrierIntervalEnd = null;
        foreach (var postdominator in impl.Blocks.Where(Item => Item != dominator &&
            StartsWithUnconditionalBarrier(Item) &&
            BarrierParametersMatch(Item, dominator) &&
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
    }

    private bool StartsWithUnconditionalBarrier(Block b)
    {
      if(b.Cmds.Count == 0) {
        return false;
      }
      CallCmd c = b.Cmds[0] as CallCmd;
      if(c == null || c.Proc != verifier.BarrierProcedure) {
        return false;
      }
      if(verifier.uniformityAnalyser.IsUniform(verifier.BarrierProcedure.Name)) {
        return true;
      }
      throw new NotImplementedException();
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

    private bool BarrierParametersMatch(Block b1, Block b2)
    {
      Debug.Assert(b1.Cmds.Count > 0);
      Debug.Assert(b2.Cmds.Count > 0);
      CallCmd barrier1 = b1.Cmds[0] as CallCmd;
      CallCmd barrier2 = b2.Cmds[0] as CallCmd;
      Debug.Assert(barrier1 != null && barrier1.Proc == verifier.BarrierProcedure);
      Debug.Assert(barrier2 != null && barrier2.Proc == verifier.BarrierProcedure);
      Debug.Assert(barrier1.Ins.Count == barrier2.Ins.Count);
      foreach(var inPair in barrier1.Ins.Zip(barrier2.Ins)) {
        if(!(inPair.Item1.ToString().Equals(inPair.Item2.ToString()))) {
          return false;
        }
      }
      return true;
    }

  }


  internal class BarrierInterval {
    private Block start;
    private Block end;
    private IEnumerable<Block> blocks;

    public BarrierInterval(Block start, Block end, DomRelation<Block> dom, DomRelation<Block> pdom, Implementation impl)
    {
      this.start = start;
      this.end = end;
      blocks = impl.Blocks.Where(Item =>
        Item != end &&
        dom.DominatedBy(Item, start) &&
        pdom.DominatedBy(Item, end)).ToList();
    }
  }

}
