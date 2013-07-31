using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

namespace GPUVerify
{
  class IntraProceduralLiveVariableAnalysis
  {
    private Graph<Block> cfg;
    private Dictionary<Block, HashSet<Variable>> liveIn;
    private Dictionary<Block, HashSet<Variable>> liveOut;

    public IntraProceduralLiveVariableAnalysis(Implementation impl)
    {
      this.cfg = Program.GraphFromImpl(impl);
    }

    public void RunAnalysis() {
      liveIn = new Dictionary<Block,HashSet<Variable>>();
      liveOut = new Dictionary<Block,HashSet<Variable>>();
      foreach(var b in cfg.Nodes) {
        liveIn[b] = new HashSet<Variable>();
        liveOut[b] = new HashSet<Variable>();
      }

      bool changed = true;
      while(changed) {
        changed = false;

        foreach(var b in cfg.Nodes) {
          var GenKillForBlock = GenKill(b);
          var GeneratedByBlock = GenKillForBlock.Item1;
          var KilledByBlock = GenKillForBlock.Item2;

          var newLiveIn = new HashSet<Variable>(liveOut[b].Where(Item => !(KilledByBlock.Contains(Item))));
          newLiveIn.UnionWith(GeneratedByBlock);

          Debug.Assert(newLiveIn.Count() >= liveIn[b].Count());
          if(newLiveIn.Count() > liveIn[b].Count()) {
            Debug.Assert(newLiveIn.Count() > liveIn[b].Count());
            liveIn[b] = newLiveIn;
            changed = true;
          }

          var newLiveOut = new HashSet<Variable>();
          foreach(var c in cfg.Successors(b)) {
            newLiveOut.UnionWith(liveIn[c]);
          }
          Debug.Assert(newLiveOut.Count() >= liveOut[b].Count());
          if(newLiveOut.Count() > liveIn[b].Count()) {
            Debug.Assert(newLiveOut.Count() > liveOut[b].Count());
            liveOut[b] = newLiveOut;
            changed = true;
          }

        }
      }

      foreach(var b in cfg.Nodes) {
        Console.WriteLine("Block: " + b.Label);
        Console.WriteLine("Live on entry: ");
        foreach(var v in liveIn[b]) {
          Console.WriteLine("  " + v.Name);
        }
        Console.WriteLine("Live on exit: ");
        foreach(var v in liveOut[b]) {
          Console.WriteLine("  " + v.Name);
        }
      }


    }

    private Dictionary<Block, Tuple<HashSet<Variable>, HashSet<Variable>>> GenKillCache = new Dictionary<Block,Tuple<HashSet<Variable>,HashSet<Variable>>>();

    private Tuple<HashSet<Variable>, HashSet<Variable>> GenKill(Block b) {
      if(!GenKillCache.ContainsKey(b)) {
        HashSet<Variable> GeneratedByBlock = new HashSet<Variable>();
        HashSet<Variable> KilledByBlock = new HashSet<Variable>();
        var result = new Tuple<HashSet<Variable>, HashSet<Variable>>(new HashSet<Variable>(), new HashSet<Variable>());
        foreach(Cmd c in b.Cmds) {
          foreach(var v in Gen(c)) {
            if(!KilledByBlock.Contains(v)) {
              GeneratedByBlock.Add(v);
            }
          }
          foreach(var v in Kill(c)) {
            KilledByBlock.Add(v);
          }
        }
        GenKillCache[b] = new Tuple<HashSet<Variable>,HashSet<Variable>>(GeneratedByBlock, KilledByBlock);
      }
      return GenKillCache[b];
    }

    private HashSet<Variable> Gen(Cmd c) {
      HashSet<Variable> result = new HashSet<Variable>();
      var assignCmd = c as AssignCmd;
      if(assignCmd != null) {
        foreach(var rhs in assignCmd.Rhss) {
          result.UnionWith(UsedVars(rhs));
        }
        foreach(var lhs in assignCmd.Rhss.OfType<MapAssignLhs>()) {
          foreach(var index in lhs.Indexes) {
            result.UnionWith(UsedVars(index));
          }
        }
        return result;
      }
      var predicateCmd = c as PredicateCmd;
      if(predicateCmd != null) {
        result.UnionWith(UsedVars(predicateCmd.Expr));
        return result;
      }
      var havocCmd = c as HavocCmd;
      if(havocCmd != null) {
        return result;
      }
      var callCmd = c as CallCmd;
      if(callCmd != null) {
        foreach(var e in callCmd.Ins) {
          result.UnionWith(UsedVars(e));
        }
        foreach(var r in callCmd.Proc.Requires) {
          result.UnionWith(UsedVars(r.Condition));
        }
        foreach(var e in callCmd.Proc.Ensures) {
          result.UnionWith(UsedVars(e.Condition));
        }
        return result;
      }
      Debug.Assert(false);
      throw new NotImplementedException();
    }

    private HashSet<Variable> Kill(Cmd c) {
      HashSet<Variable> result = new HashSet<Variable>();
      var assignCmd = c as AssignCmd;
      if(assignCmd != null) {
        foreach(var lhs in assignCmd.Lhss) {
          result.Add(lhs.DeepAssignedVariable);
        }
        return result;
      }
      var predicateCmd = c as PredicateCmd;
      if(predicateCmd != null) {
        return result;
      }
      var havocCmd = c as HavocCmd;
      if(havocCmd != null) {
        foreach(var v in havocCmd.Vars.Select(Item => Item.Decl)) {
          result.Add(v);
        }
        return result;
      }
      var callCmd = c as CallCmd;
      if(callCmd != null) {
        foreach(var v in callCmd.Outs.Select(Item => Item.Decl)) {
          result.Add(v);
        }
        foreach(IdentifierExpr ie in callCmd.Proc.Modifies) {
          result.Add(ie.Decl);
        }
        return result;
      }

      Debug.Assert(false);
      throw new NotImplementedException();
    }

    private HashSet<Variable> UsedVars(Expr e)
    {
      VariableCollector vc = new VariableCollector();
      vc.Visit(e);
      return vc.usedVars;
    }

  }
}
