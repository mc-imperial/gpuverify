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
  using System.Text;
  using System.IO;
  using System.Collections.Generic;
  using System.Text.RegularExpressions;
  using System.Linq;  
  using System.Threading;
  using System.Threading.Tasks;
  using VC;
  
  // This class allows us to parameterise each engine with specific values
  public abstract class EngineParameter
  {
    public string Name { get; set; }
  }
  
  public class EngineParameter<T> : EngineParameter
  {
    public T DefaultValue { get; set; }
    private List<T> AllowedValues;
    
    public EngineParameter(string name, T defaultValue, List<T> allowedValues = null)
    {
      this.Name = name;
      this.DefaultValue = defaultValue;
      this.AllowedValues = allowedValues;
    }
    
    public bool IsValidValue (T value)
    {
      return AllowedValues.Contains(value);
    }
  }
  
  // Abstract class from which all engines inherit.
  // Every engine maintains its own set of additional command-line parameters
  public abstract class Engine
  { 
    public int ID { get; set; }
    public bool UnderApproximating { get; set; }
    
    public Engine (int ID, bool underApproximating)
    {
      this.ID = ID;
      this.UnderApproximating = underApproximating;
    }
    
    public static List<EngineParameter> GetAllowedParameters()
    {
      return new List<EngineParameter>();
    }
    
    public static List<EngineParameter> GetRequiredParameters()
    {
      return new List<EngineParameter>();
    }
    
    public static List<Tuple<EngineParameter, EngineParameter>> GetMutuallyExclusiveParameters()
    {
      return new List<Tuple<EngineParameter, EngineParameter>>();
    }
  }
  
  // Engines based on SMT solving
  public abstract class SMTEngine : Engine
  {
    // SMT solvers
    private static string CVC4 = "cvc4";
    private static string Z3 = "z3";
    
    private static EngineParameter<string> SolverParameter;
    public static EngineParameter<string> GetSolverParameter()
    {
      if (SolverParameter == null)
        SolverParameter = new EngineParameter<string>("solver", CVC4, new List<string>{Z3, CVC4});
      return SolverParameter;
    }
    
    private static EngineParameter<int> ErrorLimitParameter;
    public static EngineParameter<int> GetErrorLimitParameter()
    {
      if (ErrorLimitParameter == null)
        ErrorLimitParameter = new EngineParameter<int>("errorlimit", 20);
      return ErrorLimitParameter;
    }    
    
    public new static List<EngineParameter> GetAllowedParameters()
    {
      return new List<EngineParameter>{ GetSolverParameter(), GetErrorLimitParameter() };
    }
    
    public Houdini.ConcurrentHoudini houdini = null;
    public string Solver { get; set; }
    public int ErrorLimit { get; set; }
    
    public SMTEngine (int ID, bool underApproximating, string solver, int errorLimit) :
      base (ID, underApproximating)
    {
	    this.Solver = solver;
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
    
    public void Start(Program program, ref Houdini.HoudiniOutcome outcome)
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
        
      Print.VerboseMessage("[CRUNCHER] Engine " + this.GetType().Name + " started");
      
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
        
      Print.VerboseMessage("[CRUNCHER] Engine " + this.GetType().Name + " finished");
      
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini) 
        OutputResults(outcome, houdiniStats);
    }
    
    // Called just before SMT cruncher starts, allowing an SMT engine to change the program if required
    public virtual void ModifyProgramBeforeCrunch (Program program)
    { 
    }
    
    private void OutputResults (Houdini.HoudiniOutcome outcome, Houdini.HoudiniSession.HoudiniStatistics houdiniStats)
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
  
  // Engine representing vanilla Houdini
  public class VanillaHoudini : SMTEngine
  {
    public static string Name = "HOUDINI";
    
    private static EngineParameter<int> DelayParameter;
    public static EngineParameter<int> GetDelayParameter()
    {
      if (DelayParameter == null)
        DelayParameter = new EngineParameter<int>("delay", 0);
      return DelayParameter;
    }
    
    private static EngineParameter<int> SlidingSecondsParameter;
    public static EngineParameter<int> GetSlidingSecondsParameter()
    {
      if (SlidingSecondsParameter == null)
        SlidingSecondsParameter = new EngineParameter<int>("slidingseconds", 0);
      return SlidingSecondsParameter;
    }
    
    private static EngineParameter<int> SlidingLimitParameter;
    public static EngineParameter<int> GetSlidingLimitParameter()
    {
      if (SlidingLimitParameter == null)
        SlidingLimitParameter = new EngineParameter<int>("slidinglimit", 1);
      return SlidingLimitParameter;
    }
    
    // Override static method from base class
    public new static List<EngineParameter> GetAllowedParameters()
    {
      List<EngineParameter> allowedParams = SMTEngine.GetAllowedParameters();
      allowedParams.Add(GetDelayParameter());
      allowedParams.Add(GetSlidingSecondsParameter());
      allowedParams.Add(GetSlidingLimitParameter());
      return allowedParams;
    }
    
    // Override static method from base class
    public new static List<Tuple<EngineParameter, EngineParameter>> GetMutuallyExclusiveParameters()
    {
      return new List<Tuple<EngineParameter, EngineParameter>> { 
        Tuple.Create<EngineParameter, EngineParameter>(GetDelayParameter(), GetSlidingSecondsParameter()),
        Tuple.Create<EngineParameter, EngineParameter>(GetDelayParameter(), GetSlidingLimitParameter())};
    }
  
    public int Delay { get; set; }
    public int SlidingSeconds { get; set; }
    public int SlidingLimit { get; set; } 
  
    public VanillaHoudini (int ID, string solver, int errorLimit):
      base(ID, false, solver, errorLimit)
    {
       this.Delay = GetDelayParameter().DefaultValue;
       this.SlidingSeconds = GetSlidingSecondsParameter().DefaultValue;
       this.SlidingLimit = GetSlidingLimitParameter().DefaultValue;
    }
    
    public override string ToString()
    {
      return Name;
    }
  }
  
  // Engine where asserts are NOT maintained in the base step
  public class SSTEP : SMTEngine
  {
    public static string Name = "SSTEP";
    
    public SSTEP (int ID, string solver, int errorLimit):
      base(ID, true, solver, errorLimit)
    {
      CommandLineOptions.Clo.Cho[this.ID].DisableLoopInvEntryAssert = true;
    }
    
    public override string ToString()
    {
      return Name;
    }
  }
  
  // Engine where asserts are NOT maintained in the induction step
  public class SBASE : SMTEngine
  {
    public static string Name = "SBASE";
    
    public SBASE (int ID, string solver, int errorLimit):
      base(ID, true, solver, errorLimit)
    {
      CommandLineOptions.Clo.Cho[this.ID].DisableLoopInvMaintainedAssert = true;
    }
    
    public override string ToString()
    {
      return Name;
    }
  }
  
  // Engine based on loop unrolling to a specific depth
  public class LU : SMTEngine
  {
    public static string Name = "LU";
    
    private static EngineParameter<int> UnrollParameter;
    public static EngineParameter<int> GetUnrollParameter()
    {
      if (UnrollParameter == null)
        UnrollParameter = new EngineParameter<int>("unroll", 1);
      return UnrollParameter;
    }   
    
    // Override static method from base class
    public new static List<EngineParameter> GetAllowedParameters()
    {
      List<EngineParameter> allowedParams = SMTEngine.GetAllowedParameters();
      allowedParams.Add(GetUnrollParameter());
      return allowedParams;
    }
    
    // Override static method from base class
    public new static List<EngineParameter> GetRequiredParameters()
    {
      List<EngineParameter> requiredParams = SMTEngine.GetRequiredParameters();
      requiredParams.Add(GetUnrollParameter());
      return requiredParams;
    }
  
    public int UnrollFactor { get; set; }
  
    public LU (int ID, string solver, int errorLimit, int unrollFactor):
      base(ID, true, solver, errorLimit)
    {
      this.UnrollFactor = unrollFactor;
    }
    
    public override void ModifyProgramBeforeCrunch (Program program)
    {
      program.UnrollLoops(this.UnrollFactor, CommandLineOptions.Clo.SoundLoopUnrolling);
    }
    
    public override string ToString()
    {
      return Name + UnrollFactor;
    }
  }
  
  // Engines based on dynamic analysis
  public class DynamicAnalysis : Engine
  {
    public static string Name = "DYNAMIC";
    
    private static EngineParameter<int> LoopHeaderLimitParameter;
    public static EngineParameter<int> GetLoopHeaderLimitParameter()
    {
      if (LoopHeaderLimitParameter == null)
        LoopHeaderLimitParameter = new EngineParameter<int>("headerlimit", 1000);
      return LoopHeaderLimitParameter;
    }
    
    private static EngineParameter<int> LoopEscapingParameter;
    public static EngineParameter<int> GetLoopEscapingParameter()
    {
      if (LoopEscapingParameter == null)
        LoopEscapingParameter = new EngineParameter<int>("loopescaping", 0);
      return LoopEscapingParameter;
    }
    
    public new static List<EngineParameter> GetAllowedParameters()
    {
      return new List<EngineParameter> { GetLoopHeaderLimitParameter(), GetLoopEscapingParameter() };
    }
    
    public int LoopHeaderLimit { get; set; }
    public int LoopEscape { get; set; }
  
    public DynamicAnalysis ():
      base(Int32.MaxValue, true)
    {
    }
    
    public BoogieInterpreter Start (Program program)
    {
      return new BoogieInterpreter(this, program);
    }
    
    public override string ToString()
    {
      return Name;
    }
  }
  
  // The pipeline of engines
  public class Pipeline
  {
    public bool Sequential { get; set;}
    public bool runHoudini { get; set; }
    
    private List<Engine> Engines = new List<Engine>();
    private int NextSMTEngineID = 0;
    private VanillaHoudini houdiniEngine = null;
    
    public Pipeline(bool sequential)
    {
      this.Sequential = sequential;
      this.runHoudini = true;
    }
    
    // Adds Houdini to the pipeline if the user has not done so
    public void AddHoudiniEngine()
    {
      foreach (Engine engine in Engines)
      {
        if (engine is VanillaHoudini)
          houdiniEngine = (VanillaHoudini) engine; 
      }
      if (houdiniEngine == null)
      {
        houdiniEngine = new VanillaHoudini(GetNextSMTEngineID(), 
                                           SMTEngine.GetSolverParameter().DefaultValue, 
                                           SMTEngine.GetErrorLimitParameter().DefaultValue);
        Engines.Add(houdiniEngine);
      }
    }
    
    public VanillaHoudini GetHoudiniEngine()
    {
      return houdiniEngine;
    }
    
    public void AddEngine(Engine engine)
    {
      this.Engines.Add(engine);
    }
    
    public int GetNextSMTEngineID()
    {
      return NextSMTEngineID++;
    }
    
    public int NumberOfSMTEngines()
    {
      return NextSMTEngineID;
    }
    
    public IEnumerable<Engine> GetEngines()
    {
      return Engines;
    }
    
    public override string ToString()
    {
      StringBuilder sb = new System.Text.StringBuilder();
      if (this.Sequential)
        sb.Append("sequential");
      else
        sb.Append("parallel");
      foreach (Engine engine in Engines)
      {
        sb.Append("-" + engine.ToString()); 
      }
      return sb.ToString();
    }
  }
  
  // The pipeline scheduler
  public class Scheduler
  {
    private List<string> FileNames;
    public int ErrorCode;
    
    public Scheduler(List<string> fileNames)
    {
      this.FileNames = fileNames;
      
      Pipeline pipeline = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).Pipeline;
      if (pipeline.runHoudini)
      {
        pipeline.AddHoudiniEngine();
      }
      
      // Execute the engine pipeline in sequence or in parallel
      if (pipeline.Sequential)
      {
        ScheduleEnginesInSequence(pipeline);
      }
      else
      {
        ScheduleEnginesInParallel(pipeline);
      }

      // If Houdini has been invoked then apply the invariants to the program.
      // Otherwise return a non-zero exit code to stop the final verification step.
      if (pipeline.runHoudini)
      {
        ErrorCode = 0;
        GPUVerify.GVUtil.IO.EmitProgram(ApplyInvariants(pipeline), GetFileNameBase(), "cbpl");          
      }
      else
      {       
        ErrorCode = 1; 
        DumpKilledInvariants(pipeline.ToString());
      }
    }
        
    private string GetFileNameBase()
    {
      string currentDir = Path.GetDirectoryName(this.FileNames[this.FileNames.Count - 1]);
      if (string.IsNullOrEmpty(currentDir))
      {
        currentDir = Directory.GetCurrentDirectory();
      }
      return currentDir + 
             Path.DirectorySeparatorChar + 
             Path.GetFileNameWithoutExtension(this.FileNames[this.FileNames.Count - 1]);      
    }
    
    private void ScheduleEnginesInSequence(Pipeline pipeline)
    {
      Houdini.HoudiniOutcome outcome = null;
      foreach (Engine engine in pipeline.GetEngines())
      {
        if (engine is SMTEngine)
        {
          SMTEngine smtEngine = (SMTEngine) engine;
          smtEngine.Start(getFreshProgram(false, false, true), ref outcome);
        }  
        else
        {
          DynamicAnalysis dynamicEngine = (DynamicAnalysis) engine;
          Program program = getFreshProgram(false, false, false);
          dynamicEngine.Start(program);
        }
      }
    }
    
    private void ScheduleEnginesInParallel(Pipeline pipeline)
    {
      Houdini.HoudiniOutcome outcome = null;
      CancellationTokenSource tokenSource = new CancellationTokenSource();
      List<Task> underApproximatingTasks = new List<Task>();
      List<Task> overApproximatingTasks = new List<Task>();
      
      // Schedule the under-approximating engines first
      foreach (Engine engine in pipeline.GetEngines())
      {
        if (!(engine is VanillaHoudini))
        {
          if (engine is DynamicAnalysis)
          {
            DynamicAnalysis dynamicEngine = (DynamicAnalysis)engine;
            underApproximatingTasks.Add(Task.Factory.StartNew(
              () =>
              {
              dynamicEngine.Start(getFreshProgram(false, false, false));
              }, 
              tokenSource.Token));
          }
          else
          {
            SMTEngine smtEngine = (SMTEngine)engine;
            underApproximatingTasks.Add(Task.Factory.StartNew(
              () =>
              {
              smtEngine.Start(getFreshProgram(false, false, true), ref outcome);
              }, 
              tokenSource.Token));
          }
        }
      }
      
      if (pipeline.runHoudini)
      {
        // We set a barrier on the under-approximating engines if a Houdini delay 
        // is specified or no sliding is selected
        if (pipeline.GetHoudiniEngine().Delay > 0)
        {
          Print.VerboseMessage("Waiting at barrier until Houdini delay has elapsed or all under-approximating engines have finished");
          Task.WaitAll(underApproximatingTasks.ToArray(), pipeline.GetHoudiniEngine().Delay * 1000);
        }
        else if (pipeline.GetHoudiniEngine().SlidingSeconds == 0)
        {
          Print.VerboseMessage("Waiting at barrier until all under-approximating engines have finished");
          Task.WaitAll(underApproximatingTasks.ToArray());
        }
       
        // Schedule the vanilla Houdini engine
        overApproximatingTasks.Add(Task.Factory.StartNew(
          () =>
          {
            pipeline.GetHoudiniEngine().Start(getFreshProgram(false, false, true), ref outcome);
          }, 
          tokenSource.Token));
              
        // Schedule Houdinis every x seconds until the number of new Houdini instances exceeds the limit
        if (pipeline.GetHoudiniEngine().SlidingSeconds > 0)
        {
          int numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;
          int newHoudinis = 0;
          bool runningHoudinis;
          
          do
          {
            // Wait before launching new Houdini instances
            Thread.Sleep((int)pipeline.GetHoudiniEngine().SlidingSeconds * 1000);
            
            // Only launch a fresh Houdini if the candidate invariant set has changed
            if (Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count > numOfRefuted)
            {
              numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;
              
              VanillaHoudini newHoudiniEngine = new VanillaHoudini(pipeline.GetNextSMTEngineID(), 
                                                  pipeline.GetHoudiniEngine().Solver, 
                                                  pipeline.GetHoudiniEngine().ErrorLimit);
              pipeline.AddEngine(newHoudiniEngine);  
              
              Print.VerboseMessage("Scheduling another Houdini instance");
              
              overApproximatingTasks.Add(Task.Factory.StartNew(
                () =>
                {
                  newHoudiniEngine.Start(getFreshProgram(false, false, true), ref outcome); 
                  tokenSource.Cancel(false);
                }, 
                tokenSource.Token));
              ++newHoudinis;
            }
            
            // Are any Houdinis still running?
            runningHoudinis = false;
            foreach (Task task in overApproximatingTasks)
            {
              if (task.Status.Equals(TaskStatus.Running))
                runningHoudinis = true;
            }
          } while (newHoudinis < pipeline.GetHoudiniEngine().SlidingLimit && runningHoudinis);
        }
        
        try
        {
          Task.WaitAny(overApproximatingTasks.ToArray(), tokenSource.Token);
          tokenSource.Cancel(false);
        }
        catch (OperationCanceledException e)
        {
          Console.WriteLine("Unexpected exception: " + e);
          throw;
        }
      }
      else
      {
        Task.WaitAll(underApproximatingTasks.ToArray());
      }
    }
    
    private Program getFreshProgram(bool raceCheck, bool divergenceCheck, bool inline)
    {
      if (!divergenceCheck)
        divergenceCheck = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).EnableBarrierDivergenceChecks;
      return GVUtil.GetFreshProgram(this.FileNames, raceCheck, divergenceCheck, inline);
    }
    
    private Program ApplyInvariants(Pipeline pipeline)
    {
      Program program = getFreshProgram(true, true, false);
      CommandLineOptions.Clo.PrintUnstructured = 2;
      pipeline.GetHoudiniEngine().houdini.ApplyAssignment(program);
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ReplaceLoopInvariantAssertions)
      {
        foreach (Block block in program.Blocks()) 
        {
          List<Cmd> newCmds = new List<Cmd>();
          foreach (Cmd cmd in block.Cmds) 
          {
            AssertCmd assertion = cmd as AssertCmd;
            if (assertion != null &&
                QKeyValue.FindBoolAttribute(assertion.Attributes,"originated_from_invariant")) 
            {
              AssumeCmd assumption = new AssumeCmd(assertion.tok, assertion.Expr, assertion.Attributes);
              newCmds.Add(assumption);
            } 
            else 
            {
              newCmds.Add(cmd);
            }
          }
          block.Cmds = newCmds;
        }
      }
      return program;
    }
    
    private void DumpKilledInvariants (string engineName)
    {
      using (StreamWriter fs = File.CreateText(GetFileNameBase() + "-killed-" + engineName + ".txt"))
      {
        foreach (string key in Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Keys) 
        {
          fs.WriteLine("FALSE: " + key);
        }
      }
    }
  }
  
}
