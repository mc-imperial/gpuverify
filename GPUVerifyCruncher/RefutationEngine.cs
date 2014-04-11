//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


using GPUVerify;

namespace Microsoft.Boogie
{
  using System;
  using System.IO;
  using System.Collections.Generic;
  using System.Text.RegularExpressions;
  using System.Linq;
  
  // Abstract class from which all engines inherit
  public abstract class Engine
  { 
    public int ID { get; set; }
    public bool UnderApproximating { get; set; }
    
    public Engine (int ID, bool underApproximating)
    {
      this.ID = ID;
      this.UnderApproximating = underApproximating;
    }
    
    public abstract void Print();
  }
  
  // Engines based on SMT solving
  public abstract class SMTEngine : Engine
  {
    public Houdini.ConcurrentHoudini houdini = null;
    
    protected static string CVC4 = "CVC4";
    protected static string Z3 = "Z3";
    
		public string Solver { get; set; }
    public int ErrorLimit { get; set; }
    
    public SMTEngine (int ID, bool underApproximating, string solver, int errorLimit) :
      base (ID, underApproximating)
    {
			this.Solver = solver.ToUpper();
      this.ErrorLimit = errorLimit;
      CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
      CommandLineOptions.Clo.Cho[this.ID].ProverCCLimit = this.ErrorLimit;
      
      foreach (string opt in CommandLineOptions.Clo.ProverOptions) 
      {
        if ((this.Solver.Equals(Z3)   && !opt.Contains("LOGIC=")) ||
            (this.Solver.Equals(CVC4) && !opt.Contains("OPTIMIZE_FOR_BV="))) 
        {
          CommandLineOptions.Clo.Cho[this.ID].ProverOptions.Add(opt);
        }
      }
    }
    
    public int Start(Program program, ref Houdini.HoudiniOutcome outcome)
    {
      if (this.Solver.Equals(CVC4))
      {
        if (CommandLineOptions.Clo.Cho[this.ID].ProverOptions.Contains("LOGIC=QF_ALL_SUPPORTED") &&
            CheckForQuantifiers.Found(program))
        {
          CommandLineOptions.Clo.Cho[this.ID].ProverOptions.Remove("LOGIC=QF_ALL_SUPPORTED");
          CommandLineOptions.Clo.Cho[this.ID].ProverOptions.Add("LOGIC=ALL_SUPPORTED");  
        }
      }
        
      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine(this.GetType().Name + " started crunching");
      
      ModifyProgramBeforeCrunch(program);
      
      Houdini.HoudiniSession.HoudiniStatistics houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
      string filename = "houdiniCexTrace_" + this.ID + ".bpl";
      houdini = new Houdini.ConcurrentHoudini(this.ID, program, houdiniStats, filename);
      
      if (outcome != null)
      {
        outcome = houdini.PerformHoudiniInference(initialAssignment: outcome.assignment);
      }
      else
      {
        outcome = houdini.PerformHoudiniInference();
      }
        
      if (CommandLineOptions.Clo.Trace) 
        Console.WriteLine(this.GetType().Name + " finished crunching");
      
      OutputResults(outcome, houdiniStats);
      
      return this.ID;
    }
    
    // Called just before SMT cruncher starts, allowing an analysis to change the program if required.
    public virtual void ModifyProgramBeforeCrunch (Program program)
    { 
    }
    
    private void OutputResults (Houdini.HoudiniOutcome outcome, Houdini.HoudiniSession.HoudiniStatistics houdiniStats)
    {
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini) 
      {
        int numTrueAssigns = outcome.assignment.Where(x => x.Value).Count();
        Console.WriteLine("Number of true assignments          = " + numTrueAssigns);
        Console.WriteLine("Number of false assignments         = " + (outcome.assignment.Count - numTrueAssigns));
        Console.WriteLine("Prover time                         = " + houdiniStats.proverTime.ToString("F2"));
        Console.WriteLine("Unsat core prover time              = " + houdiniStats.unsatCoreProverTime.ToString("F2"));
        Console.WriteLine("Number of prover queries            = " + houdiniStats.numProverQueries);
        Console.WriteLine("Number of unsat core prover queries = " + houdiniStats.numUnsatCoreProverQueries);
        Console.WriteLine("Number of unsat core prunings       = " + houdiniStats.numUnsatCorePrunings);
      }
    }
    
    public override void Print ()
    {
      Console.WriteLine("########################################");
      Console.WriteLine("# Engine:      " + this.GetType().Name);
      Console.WriteLine("# ID:          " + this.ID);
      Console.WriteLine("# Solver:      " + this.Solver);
      Console.WriteLine("# Error limit: " + this.ErrorLimit);
      Console.WriteLine("########################################");
    }
  }
  
  // Engine representing classic Houdini
  public class ClassicHoudini : SMTEngine
  {
    public ClassicHoudini (int ID, string solver = "Z3", int errorLimit = 20):
    base(ID, false, solver, errorLimit)
    {
    }
  }
  
  // Engine where asserts are NOT maintained in the base step
  public class SSTEP : SMTEngine
  {
    public SSTEP (int ID, string solver = "Z3", int errorLimit = 20):
      base(ID, true, solver, errorLimit)
    {
      CommandLineOptions.Clo.Cho[this.ID].DisableLoopInvEntryAssert = true;
    }
  }
  
  // Engine where asserts are NOT maintained in the induction step
  public class SBASE : SMTEngine
  {
    public SBASE (int ID, string solver = "Z3", int errorLimit = 20):
      base(ID, true, solver, errorLimit)
    {
      CommandLineOptions.Clo.Cho[this.ID].DisableLoopInvMaintainedAssert = true;
    }
  }
  
  // Engine based on loop unrolling to a specific depth
  public class LU : SMTEngine
  {
    public int UnrollFactor { get; set; }
  
    public LU (int ID, int unrollFactor, string solver = "Z3", int errorLimit = 20):
    base(ID, true, solver, errorLimit)
    {
      this.UnrollFactor = unrollFactor;
    }
    
    public override void ModifyProgramBeforeCrunch (Program program)
    {
      program.UnrollLoops(this.UnrollFactor, CommandLineOptions.Clo.SoundLoopUnrolling);
    }
  }
  
  // Engines based on dynamic analysis
  public class DynamicAnalysis : Engine
  {
    public DynamicAnalysis ():
    base(Int32.MaxValue, true)
    {
    }
    
    public int Start (Program program)
    {
      if (CommandLineOptions.Clo.Trace) Console.WriteLine(this.GetType().Name + " started crunching");
      new BoogieInterpreter(program);
      if (CommandLineOptions.Clo.Trace) Console.WriteLine(this.GetType().Name + " finished crunching");
      return this.ID;
    }
  
    public override void Print ()
    {
      Console.WriteLine("########################################");
      Console.WriteLine("# Engine:      " + this.GetType().Name);
      Console.WriteLine("# ID:          " + this.ID);
      Console.WriteLine("########################################");
    }
  }
  
  public class Scheduler
  {
    private List<string> FileNames;
    private SMTEngine lastEngine;
    
    public Scheduler(List<string> fileNames)
    {
      this.FileNames = fileNames;
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).SequentialPipeline.Count > 0)
      {
        ScheduleEnginesInSequence(((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).SequentialPipeline);
      }
      else
      {
        ScheduleEnginesInParallel(((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelPipeline);
      }
      
      ApplyInvariantsAndEmit();
    }
    
    private void ScheduleEnginesInSequence(List<Engine> engines)
    {
      Houdini.HoudiniOutcome outcome = null;
      foreach (Engine engine in engines)
      {
        if (engine is SMTEngine)
        {
          SMTEngine smtEngine = (SMTEngine) engine;
          smtEngine.Start(getFreshProgram(false, false, true), ref outcome);
          lastEngine = smtEngine;
        }  
        else
        {
          DynamicAnalysis dynamicEngine = (DynamicAnalysis) engine;
          dynamicEngine.Start(getFreshProgram(false, false, false));
        }
      }
    }
    
    private void ScheduleEnginesInParallel (HashSet<Engine> engines)
    {
      
    }
    
    private Program getFreshProgram(bool raceCheck, bool divergenceCheck, bool inline)
    {
      if (!divergenceCheck)
      {
        divergenceCheck = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).EnableBarrierDivergenceChecks;
      }
      return GVUtil.GetFreshProgram(this.FileNames, raceCheck, divergenceCheck, inline);
    }
    
    private void ApplyInvariantsAndEmit ()
    {
      List<string> filesToProcess = new List<string>();
      filesToProcess.Add(this.FileNames[this.FileNames.Count - 1]);
      string directoryContainingFiles = Path.GetDirectoryName (filesToProcess [0]);
      if (string.IsNullOrEmpty (directoryContainingFiles))
        directoryContainingFiles = Directory.GetCurrentDirectory ();
      var annotatedFile = directoryContainingFiles + Path.DirectorySeparatorChar +
        Path.GetFileNameWithoutExtension(filesToProcess[0]);

      Program program = getFreshProgram(true, true, false);
      CommandLineOptions.Clo.PrintUnstructured = 2;
      
      lastEngine.houdini.ApplyAssignment(program);
      
      GPUVerify.GVUtil.IO.EmitProgram(program, annotatedFile, "cbpl");
    }
  }

  /// <summary>
  /// Wrapper class for a concurrent Houdini instance. It is able to run either on
  /// the main thread or on a worker thread.
  /// </summary>
  public class StaticRefutationEngine : RefutationEngine
  {
    private Houdini.ConcurrentHoudini houdini = null;
    private bool isTrusted;
    private string solver;
    private int errorLimit;
    private bool disableLEI;
    private bool disableLMI;
    private bool modifyTSO;
    private int loopUnroll;

    public override int ID { get { return this.id; } }
    public override string Name { get { return this.name; } }
    public bool IsTrusted { get { return this.isTrusted; } }
    public Houdini.ConcurrentHoudini Houdini { get { return this.houdini; } }

    public StaticRefutationEngine(int id, string name, string errorLimit, string disableLEI,
                                  string disableLMI, string modifyTSO, string loopUnroll)
    {
      this.id = id;
      this.name = name;
      this.errorLimit = int.Parse(errorLimit);
      this.disableLEI = bool.Parse(disableLEI);
      this.disableLMI = bool.Parse(disableLMI);
      this.modifyTSO = bool.Parse(modifyTSO);
      this.loopUnroll = int.Parse(loopUnroll);

      CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
      CommandLineOptions.Clo.Cho[id].ProverCCLimit = this.errorLimit;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvEntryAssert = this.disableLEI;
      CommandLineOptions.Clo.Cho[id].DisableLoopInvMaintainedAssert = this.disableLMI;
      CommandLineOptions.Clo.Cho[id].ModifyTopologicalSorting = this.modifyTSO;

      if (CommandLineOptions.Clo.ProverOptions.Contains("SOLVER=cvc4"))
        this.solver = "cvc4";
      else
        this.solver = "Z3";

      foreach (var opt in CommandLineOptions.Clo.ProverOptions) {
        if ((this.solver.Equals("Z3") && !opt.Contains("LOGIC=")) ||
            (this.solver.Equals("cvc4") && !opt.Contains("OPTIMIZE_FOR_BV="))) {
          CommandLineOptions.Clo.Cho[id].ProverOptions.Add(opt);
        }
      }

      if (this.disableLEI || this.disableLMI || this.loopUnroll != -1)
        this.isTrusted = false;
      else
        this.isTrusted = true;

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DynamicErrorLimit > 0 && this.isTrusted)
        CommandLineOptions.Clo.Cho[id].ProverCCLimit = 1;
    }

    /// <summary>
    /// Starts a new concurrent Houdini execution. Returns the outcome of the
    /// Houdini process by reference.
    /// </summary>
    public int start(Program program, ref Houdini.HoudiniOutcome outcome)
    {
      if (solver.Equals("cvc4"))
        KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(program, id);

      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] started crunching ...");
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo)
        printConfig();

      if (loopUnroll != -1)
        program.UnrollLoops(loopUnroll, CommandLineOptions.Clo.SoundLoopUnrolling);

      var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
      houdini = new Houdini.ConcurrentHoudini(id, program, houdiniStats, "houdiniCexTrace_" + id +".bpl");

      if (outcome != null)
        outcome = houdini.PerformHoudiniInference(initialAssignment: outcome.assignment);
      else
        outcome = houdini.PerformHoudiniInference();

      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] finished.");

      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini) {
        int numTrueAssigns = 0;
        foreach (var x in outcome.assignment) {
          if (x.Value)
            numTrueAssigns++;
        }

        Console.WriteLine("Number of true assignments = " + numTrueAssigns);
        Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));
        Console.WriteLine("Prover time = " + houdiniStats.proverTime.ToString("F2"));
        Console.WriteLine("Unsat core prover time = " + houdiniStats.unsatCoreProverTime.ToString("F2"));
        Console.WriteLine("Number of prover queries = " + houdiniStats.numProverQueries);
        Console.WriteLine("Number of unsat core prover queries = " + houdiniStats.numUnsatCoreProverQueries);
        Console.WriteLine("Number of unsat core prunings = " + houdiniStats.numUnsatCorePrunings);
      }

      return id;
    }

    /// <summary>
    /// Prints the configuration options of the Static Refutation Engine.
    /// </summary>
    public override void printConfig()
    {
      Console.WriteLine("######################################");
      Console.WriteLine("# Configuration for " + name + ":");
      Console.WriteLine("# id = " + id);
      Console.WriteLine("# solver = " + solver);
      Console.WriteLine("# errorLimit = " + errorLimit);
      Console.WriteLine("# disableLEI = " + disableLEI);
      Console.WriteLine("# disableLMI = " + disableLMI);
      Console.WriteLine("# modifyTSO = " + modifyTSO);
      Console.WriteLine("# loopUnroll = " + loopUnroll);
      Console.WriteLine("######################################");
    }
  }

  /// <summary>
  /// Wrapper class for a concurrent dynamic analyser instance. It is able to run either on
  /// the main thread or on a worker thread.
  /// </summary>
  public class DynamicRefutationEngine : RefutationEngine
  {
    private int threadId_X;
    private int threadId_Y;
    private int threadId_Z;
    private int groupId_X;
    private int groupId_Y;
    private int groupId_Z;

    public override int ID { get { return this.id; } }
    public override string Name { get { return this.name; } }
    
    public BoogieInterpreter Interpreter;

    public DynamicRefutationEngine(int id, string name, string threadId_X, string threadId_Y, string threadId_Z,
                                   string groupId_X, string groupId_Y, string groupId_Z)
    {
      this.id = id;
      this.name = name;
      this.threadId_X = checkForMaxAndParseInt(threadId_X);
      this.threadId_Y = checkForMaxAndParseInt(threadId_Y);
      this.threadId_Z = checkForMaxAndParseInt(threadId_Z);
      this.groupId_X = checkForMaxAndParseInt(groupId_X);
      this.groupId_Y = checkForMaxAndParseInt(groupId_Y);
      this.groupId_Z = checkForMaxAndParseInt(groupId_Z);
    }

    /// <summary>
    /// Starts a new concurrent dynamic analyser execution.
    /// </summary>
    public void start(Program program, bool verbose = false, int debug = 0)
    {
      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] started crunching ...");
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).InferInfo)
        printConfig();

      Interpreter = new BoogieInterpreter(program);

      if (CommandLineOptions.Clo.Trace)
        Console.WriteLine("INFO:[Engine-" + name + "] finished.");
    }

    private int checkForMaxAndParseInt(string value)
    {
      if (value.ToLower().Equals("max")) return int.MaxValue;
      else return int.Parse(value);
    }

    /// <summary>
        /// Prints the configuration options of the Dynamic Refutation Engine.
    /// </summary>
    public override void printConfig()
    {
      Console.WriteLine("######################################");
      Console.WriteLine("# Configuration for " + name + ":");
      Console.WriteLine("# id = " + id);
      Console.WriteLine("# threadId_X = " + threadId_X);
      Console.WriteLine("# threadId_Y = " + threadId_Y);
      Console.WriteLine("# threadId_Z = " + threadId_Z);
      Console.WriteLine("# groupId_X = " + groupId_X);
      Console.WriteLine("# groupId_Y = " + groupId_Y);
      Console.WriteLine("# groupId_Z = " + groupId_Z);
      Console.WriteLine("######################################");
    }
  }

  /// <summary>
  /// An abstract class for a refutation engine.
  /// </summary>
  public abstract class RefutationEngine
  {
    protected int id;
    protected string name;

    public abstract int ID { get; }
    public abstract string Name { get; }

    /// <summary>
    /// Prints the configuration options of the Refutation Engine.
    /// </summary>
    public abstract void printConfig();
  }
}
