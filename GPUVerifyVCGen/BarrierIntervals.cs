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
    }

    private HashSet<BarrierInterval> ComputeBarrierIntervals(Implementation impl)
    {
      HashSet<BarrierInterval> result = new HashSet<BarrierInterval>();

      ExtractCommandsIntoBlocks(impl, Item => (Item is CallCmd && GPUVerifier.IsBarrier(((CallCmd)Item).Proc)));
      Graph<Block> cfg = Program.GraphFromImpl(impl);

      // If the CFG has no exit nodes, i.e. it cannot terminate,
      // we bail out; we need a single-entry single-exit CFG
      // and we cannot get one under such circumstances
      if (NoExitFromCFG(cfg))
      {
        return result;
      }

      // To make the CFG single-exit, we add a special exit block
      Block SpecialExitBlock = new Block();
      cfg.Nodes.Add(SpecialExitBlock);

      // Now link any existing CFG node that has no successors to the
      // special exit node.
      foreach (var b in cfg.Nodes)
      {
        if (b == SpecialExitBlock)
        {
          continue;
        }
        if (cfg.Successors(b).Count() == 0)
        {
          cfg.AddEdge(b, SpecialExitBlock);
        }
      }

      Graph<Block> dual = cfg.Dual(new Block());
      DomRelation<Block> dom = cfg.DominatorMap;
      DomRelation<Block> pdom = dual.DominatorMap;

      foreach (var dominator in cfg.Nodes.Where(Item => StartsWithUnconditionalBarrier(Item, impl, SpecialExitBlock)))
      {
        Block smallestBarrierIntervalEnd = null;
        foreach (var postdominator in cfg.Nodes.Where(Item => Item != dominator &&
            StartsWithUnconditionalBarrier(Item, impl, SpecialExitBlock) &&
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
      if (GPUVerifyVCGenCommandLineOptions.DebugGPUVerify)
      {
        Console.WriteLine("Found " + result.Count() + " barrier interval(s) in " + impl.Name);
      }
      return result;
    }

    private static bool NoExitFromCFG(Graph<Block> cfg)
    {
      foreach (var b in cfg.Nodes)
      {
        if (cfg.Successors(b).Count() == 0)
        {
          return false;
        }
      }
      return true;
    }

    private bool StartsWithUnconditionalBarrier(Block b, Implementation Impl, Block SpecialExitBlock)
    {
      if (verifier.IsKernelProcedure(Impl.Proc))
      {
        if (b == Impl.Blocks[0]) {
          // There is a barrier at the very start of the kernel
          return true;
        }
        if (b == SpecialExitBlock)
        {
          // There is a barrier at the very end of the kernel
          return true;
        }
      }

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

    private static List<List<Cmd>> PartitionCmdsAccordingToPredicate(List<Cmd> Cmds, Func<Cmd, bool> Predicate) {
      List<List<Cmd>> result = new List<List<Cmd>>();
      List<Cmd> current = new List<Cmd>();
      result.Add(current);
      foreach(Cmd cmd in Cmds) {
        if(Predicate(cmd) && current.Count > 0) {
           current = new List<Cmd>();
           result.Add(current);
        }
        current.Add(cmd);
      }
      return result;
    }

    void ExtractCommandsIntoBlocks(Implementation impl, Func<Cmd, bool> Predicate) {
      Dictionary<Block, Block> oldToNew = new Dictionary<Block,Block>();
      HashSet<Block> newBlocks = new HashSet<Block>();
      HashSet<Block> removedBlocks = new HashSet<Block>();
      Block newEntryBlock = null;

      foreach (Block b in impl.Blocks)
      {
        List<List<Cmd>> partition = PartitionCmdsAccordingToPredicate(b.Cmds, Predicate);
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
          verifier.KernelArrayInfo.GetGroupSharedArrays(true).Where(Item =>
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
    private IEnumerable<Block> blocks;

    public IEnumerable<Block> Blocks {
      get { return blocks; }
    }

    public BarrierInterval(Block start, Block end, DomRelation<Block> dom, DomRelation<Block> pdom, Implementation impl)
    {
      blocks = impl.Blocks.Where(Item =>
        Item != end &&
        dom.DominatedBy(Item, start) &&
        pdom.DominatedBy(Item, end)).ToList();
    }

    internal HashSet<Variable> FindWrittenGroupSharedArrays(GPUVerifier verifier)
    {

      // We add any group-shared array that may be written to or accessed atomically
      // in the region.
      //
      // We also add any group-shared array that may be written to by an asynchronous
      // memory copy somewhere in the kernel.  This is because asynchronous copies can
      // cross barriers.  Currently we are very conservative about this.

      HashSet<Variable> result = new HashSet<Variable>();

      foreach (var v in verifier.KernelArrayInfo.GetGroupSharedArrays(false))
      {
        if (verifier.ArraysAccessedByAsyncWorkGroupCopy[AccessType.WRITE].Contains(v.Name))
        {
          result.Add(v);
        }
      }

      foreach (var m in Blocks.Select(Item => Item.Cmds).SelectMany(Item => Item).OfType<CallCmd>()
        .Select(Item => Item.Proc.Modifies).SelectMany(Item => Item))
      {
        // m is a variable modified by a call in the barrier interval
        Variable v;
        if(verifier.TryGetArrayFromAccessHasOccurred(GVUtil.StripThreadIdentifier(m.Name), AccessType.WRITE, out v) ||
           verifier.TryGetArrayFromAccessHasOccurred(GVUtil.StripThreadIdentifier(m.Name), AccessType.ATOMIC, out v)) {
          if (verifier.KernelArrayInfo.GetGroupSharedArrays(false).Contains(v)) {
            result.Add(v);
          }
        }
      }
      return result;
    }

  }

}
