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
    private Implementation impl;
    private Dictionary<string, LocalDescriptor> locals;
    private Dictionary<string, GlobalDescriptor> globals;
    private Dictionary<Block, HashSet<VariableDescriptor>> liveIn;
    private Dictionary<Block, HashSet<VariableDescriptor>> liveOut;

    public IntraProceduralLiveVariableAnalysis(Program prog, Implementation impl)
    {
      this.impl = impl;
      locals = new Dictionary<string,LocalDescriptor>();
      foreach(var v in impl.InParams) {
        locals[v.Name] = new LocalDescriptor(impl.Name, v.Name);
      }
      foreach(var v in impl.OutParams) {
        locals[v.Name] = new LocalDescriptor(impl.Name, v.Name);
      }
      foreach(var v in impl.LocVars) {
        locals[v.Name] = new LocalDescriptor(impl.Name, v.Name);
      }
      globals = new Dictionary<string,GlobalDescriptor>();
      foreach(var v in prog.TopLevelDeclarations.OfType<Variable>()) {
        globals[v.Name] = new GlobalDescriptor(v.Name);
      }
    }

    public void RunAnalysis() {
      liveIn = new Dictionary<Block,HashSet<VariableDescriptor>>();
      liveOut = new Dictionary<Block,HashSet<VariableDescriptor>>();
      Graph<Block> cfg = Program.GraphFromImpl(impl);

      foreach(var b in cfg.Nodes) {
        liveIn[b] = new HashSet<VariableDescriptor>();
        liveOut[b] = new HashSet<VariableDescriptor>();
      }

      bool changed = true;
      while(changed) {
        changed = false;

        foreach(var b in cfg.Nodes) {
          var GenKillForBlock = GenKill(b);
          var GeneratedByBlock = GenKillForBlock.Item1;
          var KilledByBlock = GenKillForBlock.Item2;

          var newLiveIn = new HashSet<VariableDescriptor>(liveOut[b].Where(Item => !(KilledByBlock.Contains(Item))));
          newLiveIn.UnionWith(GeneratedByBlock);

          Debug.Assert(newLiveIn.Count() >= liveIn[b].Count());
          if(newLiveIn.Count() > liveIn[b].Count()) {
            Debug.Assert(newLiveIn.Count() > liveIn[b].Count());
            liveIn[b] = newLiveIn;
            changed = true;
          }

          var newLiveOut = new HashSet<VariableDescriptor>();
          foreach(var c in cfg.Successors(b)) {
            newLiveOut.UnionWith(liveIn[c]);
          }
          Debug.Assert(newLiveOut.Count() >= liveOut[b].Count());
          if(newLiveOut.Count() > liveOut[b].Count()) {
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
          Console.WriteLine("  " + v);
        }
        Console.WriteLine("Live on exit: ");
        foreach(var v in liveOut[b]) {
          Console.WriteLine("  " + v);
        }
      }


    }

    private Dictionary<Block, Tuple<HashSet<VariableDescriptor>, HashSet<VariableDescriptor>>> GenKillCache = 
      new Dictionary<Block,Tuple<HashSet<VariableDescriptor>,HashSet<VariableDescriptor>>>();

    private Tuple<HashSet<VariableDescriptor>, HashSet<VariableDescriptor>> GenKill(Block b) {
      if(!GenKillCache.ContainsKey(b)) {
        HashSet<VariableDescriptor> GeneratedByBlock = new HashSet<VariableDescriptor>();
        HashSet<VariableDescriptor> KilledByBlock = new HashSet<VariableDescriptor>();
        var result = new Tuple<HashSet<VariableDescriptor>, HashSet<VariableDescriptor>>(new HashSet<VariableDescriptor>(), new HashSet<VariableDescriptor>());
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
        GenKillCache[b] = new Tuple<HashSet<VariableDescriptor>,HashSet<VariableDescriptor>>(GeneratedByBlock, KilledByBlock);
      }
      return GenKillCache[b];
    }

    private HashSet<VariableDescriptor> Gen(Cmd c) {
      HashSet<VariableDescriptor> result = new HashSet<VariableDescriptor>();
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

    private HashSet<VariableDescriptor> Kill(Cmd c) {
      HashSet<VariableDescriptor> result = new HashSet<VariableDescriptor>();
      var assignCmd = c as AssignCmd;
      if(assignCmd != null) {
        foreach(var lhs in assignCmd.Lhss) {
          result.Add(MakeDescriptor(lhs.DeepAssignedVariable));
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
          result.Add(MakeDescriptor(v));
        }
        return result;
      }
      var callCmd = c as CallCmd;
      if(callCmd != null) {
        foreach(var v in callCmd.Outs.Select(Item => Item.Decl)) {
          result.Add(MakeDescriptor(v));
        }
        foreach(IdentifierExpr ie in callCmd.Proc.Modifies) {
          result.Add(MakeDescriptor(ie.Decl, true));
        }
        return result;
      }

      Debug.Assert(false);
      throw new NotImplementedException();
    }

    private HashSet<VariableDescriptor> UsedVars(Expr e)
    {
      VariableCollector vc = new VariableCollector();
      vc.Visit(e);
      HashSet<VariableDescriptor> result = new HashSet<VariableDescriptor>();
      foreach(var v in vc.usedVars) {
        var vd = MakeDescriptor(v);
        if(vd != null) {
          result.Add(vd);
        }
      }
      return result;
    }

    private VariableDescriptor MakeDescriptor(Variable v, bool isGlobal = false) {
      if(locals.ContainsKey(v.Name) && !isGlobal) {
        return locals[v.Name];
      }
      if(globals.ContainsKey(v.Name)) {
        return globals[v.Name];
      }
      return null;
    }

  }
}
