//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace Microsoft.Boogie
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Text;
    using System.Threading;
    using System.Threading.Tasks;
    using GPUVerify;

    // This class allows us to parameterise each engine with specific values
    // Abstract class from which all engines inherit.
    // Every engine maintains its own set of additional command-line parameters
    public abstract class RefutationEngine
    {
        public int ID { get; set; }

        public bool UnderApproximating { get; set; }

        public RefutationEngine(int id, bool underApproximating)
        {
            ID = id;
            UnderApproximating = underApproximating;
        }

        public static List<Parameter> GetAllowedParameters()
        {
            return new List<Parameter>();
        }

        public static List<Parameter> GetRequiredParameters()
        {
            return new List<Parameter>();
        }

        public static List<Tuple<Parameter, Parameter>> GetMutuallyExclusiveParameters()
        {
            return new List<Tuple<Parameter, Parameter>>();
        }

        public abstract class Parameter
        {
            public string Name { get; }

            protected Parameter(string name)
            {
                Name = name;
            }
        }

        public class Parameter<T> : Parameter
        {
            public T DefaultValue { get; }

            private List<T> allowedValues;

            public Parameter(string name, T defaultValue, List<T> allowedValues = null)
                : base(name)
            {
                this.DefaultValue = defaultValue;
                this.allowedValues = allowedValues;
            }

            public bool IsValidValue(T value)
            {
                return allowedValues.Contains(value);
            }
        }
    }

    // Engines based on SMT solving
    public abstract class SMTEngine : RefutationEngine
    {
        // SMT solvers
        private const string CVC4 = "cvc4";
        private const string Z3 = "z3";

        private static Parameter<string> solverParameter;

        public static Parameter<string> GetSolverParameter()
        {
            if (solverParameter == null)
                solverParameter = new Parameter<string>("solver", CVC4, new List<string> { Z3, CVC4 });
            return solverParameter;
        }

        private static Parameter<int> errorLimitParameter;

        public static Parameter<int> GetErrorLimitParameter()
        {
            if (errorLimitParameter == null)
                errorLimitParameter = new Parameter<int>("errorlimit", 20);
            return errorLimitParameter;
        }

        public static new List<Parameter> GetAllowedParameters()
        {
            return new List<Parameter> { GetSolverParameter(), GetErrorLimitParameter() };
        }

        private Houdini.ConcurrentHoudini houdini = null;

        public string Solver { get; set; }

        public int ErrorLimit { get; set; }

        public SMTEngine(int id, bool underApproximating, string solver, int errorLimit)
            : base(id, underApproximating)
        {
            Solver = solver;
            ErrorLimit = errorLimit;
            CommandLineOptions.Clo.Cho.Add(new CommandLineOptions.ConcurrentHoudiniOptions());
            CommandLineOptions.Clo.Cho[ID].ProverCCLimit = ErrorLimit;

            foreach (string opt in CommandLineOptions.Clo.ProverOptions)
            {
                if ((Solver.Equals(Z3) && !opt.Contains("LOGIC="))
                    || (Solver.Equals(CVC4) && !opt.Contains("OPTIMIZE_FOR_BV=")))
                {
                    CommandLineOptions.Clo.Cho[ID].ProverOptions.Add(opt);
                }
            }
        }

        public void Start(Program program, ref Houdini.HoudiniOutcome outcome)
        {
            if (Solver.Equals(CVC4))
            {
                if (CommandLineOptions.Clo.Cho[ID].ProverOptions.Contains("LOGIC=QF_ALL_SUPPORTED")
                    && CheckForQuantifiersVisitor.Find(program))
                {
                    CommandLineOptions.Clo.Cho[ID].ProverOptions.Remove("LOGIC=QF_ALL_SUPPORTED");
                    CommandLineOptions.Clo.Cho[ID].ProverOptions.Add("LOGIC=ALL_SUPPORTED");
                }
            }

            Print.VerboseMessage("[CRUNCHER] Engine " + GetType().Name + " started");

            ModifyProgramBeforeCrunch(program);

            Houdini.HoudiniSession.HoudiniStatistics houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();

            if (CommandLineOptions.Clo.StagedHoudini != null)
            {
                // This is an initial attempt at hooking GPUVerify up with Staged Houdini.
                // More work is required to make this compatible with other cruncher options,
                // and we start with a couple of crude hacks to work around the fact that
                // Staged Houdini is not integrated with ConcurrentHoudini.
                CommandLineOptions.Clo.ConcurrentHoudini = false; // HACK - requires proper integration
                CommandLineOptions.Clo.ProverCCLimit = ErrorLimit; // HACK - requires proper integration
                Debug.Assert(outcome == null);
                Debug.Assert(this is VanillaHoudini);

                Houdini.StagedHoudini houdini = new Houdini.StagedHoudini(program, houdiniStats, ExecutionEngine.ProgramFromFile);

                outcome = houdini.PerformStagedHoudiniInference();
            }
            else
            {
                string filename = "houdiniCexTrace_" + ID + ".bpl";
                houdini = new Houdini.ConcurrentHoudini(ID, program, houdiniStats, filename);

                if (outcome != null)
                {
                    outcome = houdini.PerformHoudiniInference(initialAssignment: outcome.assignment);
                }
                else
                {
                    outcome = houdini.PerformHoudiniInference();
                }
            }

            Print.VerboseMessage("[CRUNCHER] Engine " + GetType().Name + " finished");

            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).DebugConcurrentHoudini)
                OutputResults(outcome, houdiniStats);
        }

        // Called just before SMT cruncher starts, allowing an SMT engine to change the program if required
        public virtual void ModifyProgramBeforeCrunch(Program program)
        {
        }

        private void OutputResults(Houdini.HoudiniOutcome outcome, Houdini.HoudiniSession.HoudiniStatistics houdiniStats)
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
        public const string Name = "HOUDINI";

        private static Parameter<int> delayParameter;

        public static Parameter<int> GetDelayParameter()
        {
            if (delayParameter == null)
                delayParameter = new Parameter<int>("delay", 0);

            return delayParameter;
        }

        private static Parameter<int> slidingSecondsParameter;

        public static Parameter<int> GetSlidingSecondsParameter()
        {
            if (slidingSecondsParameter == null)
                slidingSecondsParameter = new Parameter<int>("slidingseconds", 0);

            return slidingSecondsParameter;
        }

        private static Parameter<int> slidingLimitParameter;

        public static Parameter<int> GetSlidingLimitParameter()
        {
            if (slidingLimitParameter == null)
                slidingLimitParameter = new Parameter<int>("slidinglimit", 1);

            return slidingLimitParameter;
        }

        // Override static method from base class
        public static new List<Parameter> GetAllowedParameters()
        {
            List<Parameter> allowedParams = SMTEngine.GetAllowedParameters();
            allowedParams.Add(GetDelayParameter());
            allowedParams.Add(GetSlidingSecondsParameter());
            allowedParams.Add(GetSlidingLimitParameter());
            return allowedParams;
        }

        // Override static method from base class
        public static new List<Tuple<Parameter, Parameter>> GetMutuallyExclusiveParameters()
        {
            return new List<Tuple<Parameter, Parameter>>
            {
                Tuple.Create<Parameter, Parameter>(GetDelayParameter(), GetSlidingSecondsParameter()),
                Tuple.Create<Parameter, Parameter>(GetDelayParameter(), GetSlidingLimitParameter())
            };
        }

        public int Delay { get; set; }

        public int SlidingSeconds { get; set; }

        public int SlidingLimit { get; set; }

        public VanillaHoudini(int id, string solver, int errorLimit)
            : base(id, false, solver, errorLimit)
        {
            Delay = GetDelayParameter().DefaultValue;
            SlidingSeconds = GetSlidingSecondsParameter().DefaultValue;
            SlidingLimit = GetSlidingLimitParameter().DefaultValue;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    // Engine where asserts are NOT maintained in the base step
    public class SSTEP : SMTEngine
    {
        public const string Name = "SSTEP";

        public SSTEP(int id, string solver, int errorLimit)
            : base(id, true, solver, errorLimit)
        {
            CommandLineOptions.Clo.Cho[ID].DisableLoopInvEntryAssert = true;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    // Engine where asserts are NOT maintained in the induction step
    public class SBASE : SMTEngine
    {
        public const string Name = "SBASE";

        public SBASE(int id, string solver, int errorLimit)
            : base(id, true, solver, errorLimit)
        {
            CommandLineOptions.Clo.Cho[ID].DisableLoopInvMaintainedAssert = true;
        }

        public override string ToString()
        {
            return Name;
        }
    }

    // Engine based on loop unrolling to a specific depth
    public class LU : SMTEngine
    {
        public const string Name = "LU";

        private static Parameter<int> unrollParameter;

        public static Parameter<int> GetUnrollParameter()
        {
            if (unrollParameter == null)
                unrollParameter = new Parameter<int>("unroll", 1);
            return unrollParameter;
        }

        // Override static method from base class
        public static new List<Parameter> GetAllowedParameters()
        {
            List<Parameter> allowedParams = SMTEngine.GetAllowedParameters();
            allowedParams.Add(GetUnrollParameter());
            return allowedParams;
        }

        // Override static method from base class
        public static new List<Parameter> GetRequiredParameters()
        {
            List<Parameter> requiredParams = SMTEngine.GetRequiredParameters();
            requiredParams.Add(GetUnrollParameter());
            return requiredParams;
        }

        public int UnrollFactor { get; set; }

        public LU(int id, string solver, int errorLimit, int unrollFactor)
            : base(id, true, solver, errorLimit)
        {
            UnrollFactor = unrollFactor;
        }

        public override void ModifyProgramBeforeCrunch(Program program)
        {
            program.UnrollLoops(UnrollFactor, CommandLineOptions.Clo.SoundLoopUnrolling);
        }

        public override string ToString()
        {
            return Name + UnrollFactor;
        }
    }

    // Engines based on dynamic analysis
    public class DynamicAnalysis : RefutationEngine
    {
        public const string Name = "DYNAMIC";

        private static Parameter<int> loopHeaderLimitParameter;

        public static Parameter<int> GetLoopHeaderLimitParameter()
        {
            if (loopHeaderLimitParameter == null)
                loopHeaderLimitParameter = new Parameter<int>("headerlimit", 1000);

            return loopHeaderLimitParameter;
        }

        private static Parameter<int> loopEscapingParameter;

        public static Parameter<int> GetLoopEscapingParameter()
        {
            if (loopEscapingParameter == null)
                loopEscapingParameter = new Parameter<int>("loopescaping", 0);

            return loopEscapingParameter;
        }

        private static Parameter<int> timeLimitParameter;

        public static Parameter<int> GetTimeLimitParameter()
        {
            if (timeLimitParameter == null)
                timeLimitParameter = new Parameter<int>("timelimit", int.MaxValue);

            return timeLimitParameter;
        }

        public static new List<Parameter> GetAllowedParameters()
        {
            return new List<Parameter> { GetLoopHeaderLimitParameter(), GetLoopEscapingParameter(), GetTimeLimitParameter() };
        }

        public int LoopHeaderLimit { get; set; }

        public int LoopEscape { get; set; }

        public int TimeLimit { get; set; }

        public DynamicAnalysis()
            : base(int.MaxValue, true)
        {
        }

        public BoogieInterpreter Start(Program program)
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
        public bool Sequential { get; set; }

        public bool RunHoudini { get; set; }

        private List<RefutationEngine> engines = new List<RefutationEngine>();
        private int nextSMTEngineID = 0;
        private VanillaHoudini houdiniEngine = null;

        public Pipeline(bool sequential)
        {
            Sequential = sequential;
            RunHoudini = true;
        }

        // Adds Houdini to the pipeline if the user has not done so
        public void AddHoudiniEngine()
        {
            foreach (RefutationEngine engine in engines)
            {
                if (engine is VanillaHoudini)
                    houdiniEngine = (VanillaHoudini)engine;
            }

            if (houdiniEngine == null)
            {
                houdiniEngine = new VanillaHoudini(
                    GetNextSMTEngineID(),
                    SMTEngine.GetSolverParameter().DefaultValue,
                    SMTEngine.GetErrorLimitParameter().DefaultValue);
                engines.Add(houdiniEngine);
            }
        }

        public VanillaHoudini GetHoudiniEngine()
        {
            return houdiniEngine;
        }

        public void AddEngine(RefutationEngine engine)
        {
            engines.Add(engine);
        }

        public int GetNextSMTEngineID()
        {
            return nextSMTEngineID++;
        }

        public int NumberOfSMTEngines()
        {
            return nextSMTEngineID;
        }

        public IEnumerable<RefutationEngine> GetEngines()
        {
            return engines;
        }

        public override string ToString()
        {
            StringBuilder sb = new System.Text.StringBuilder();
            if (Sequential)
                sb.Append("sequential");
            else
                sb.Append("parallel");

            foreach (RefutationEngine engine in engines)
                sb.Append("-" + engine.ToString());

            return sb.ToString();
        }
    }

    // The pipeline scheduler
    public class Scheduler
    {
        private List<string> fileNames;

        public int ErrorCode { get; private set; }

        public Scheduler(List<string> fileNames)
        {
            this.fileNames = fileNames;

            Pipeline pipeline = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).Pipeline;
            if (pipeline.RunHoudini)
                pipeline.AddHoudiniEngine();

            Houdini.HoudiniOutcome outcome;

            // Execute the engine pipeline in sequence or in parallel
            if (pipeline.Sequential)
                outcome = ScheduleEnginesInSequence(pipeline);
            else
                outcome = ScheduleEnginesInParallel(pipeline);

            // If Houdini has been invoked then apply the invariants to the program.
            // Otherwise report success
            if (pipeline.RunHoudini)
            {
                var counters = default(KernelAnalyser.ResultCounter);
                Program prog = GetFreshProgram(true, true);
                foreach (var implOutcome in outcome.implementationOutcomes)
                {
                    KernelAnalyser.ProcessOutcome(
                        prog, implOutcome.Key, implOutcome.Value.outcome, implOutcome.Value.errors, string.Empty, ref counters);
                }

                if (counters.AllVerified())
                    GPUVerify.Utilities.IO.EmitProgram(ApplyInvariants(outcome), GetFileNameBase(), "cbpl");

                ErrorCode = KernelAnalyser.GetExitCode(counters);
            }
            else
            {
                ErrorCode = (int)ToolExitCodes.SUCCESS;
            }

            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).WriteKilledInvariantsToFile)
            {
                DumpKilledInvariants(pipeline.ToString());
            }
        }

        private string GetFileNameBase()
        {
            string currentDir = Path.GetDirectoryName(fileNames[fileNames.Count - 1]);
            if (string.IsNullOrEmpty(currentDir))
            {
                currentDir = Directory.GetCurrentDirectory();
            }

            return currentDir +
                   Path.DirectorySeparatorChar +
                   Path.GetFileNameWithoutExtension(fileNames[fileNames.Count - 1]);
        }

        private Houdini.HoudiniOutcome ScheduleEnginesInSequence(Pipeline pipeline)
        {
            Houdini.HoudiniOutcome outcome = null;
            foreach (RefutationEngine engine in pipeline.GetEngines())
            {
                if (engine is SMTEngine)
                {
                    SMTEngine smtEngine = (SMTEngine)engine;
                    smtEngine.Start(GetFreshProgram(true, true), ref outcome);
                }
                else
                {
                    DynamicAnalysis dynamicEngine = (DynamicAnalysis)engine;
                    Program program = GetFreshProgram(true, false);
                    dynamicEngine.Start(program);
                }
            }

            return outcome;
        }

        private Houdini.HoudiniOutcome ScheduleEnginesInParallel(Pipeline pipeline)
        {
            Houdini.HoudiniOutcome outcome = null;
            CancellationTokenSource tokenSource = new CancellationTokenSource();
            List<Task> underApproximatingTasks = new List<Task>();
            List<Task> overApproximatingTasks = new List<Task>();

            // Schedule the under-approximating engines first
            foreach (RefutationEngine engine in pipeline.GetEngines())
            {
                if (!(engine is VanillaHoudini))
                {
                    if (engine is DynamicAnalysis)
                    {
                        DynamicAnalysis dynamicEngine = (DynamicAnalysis)engine;
                        underApproximatingTasks.Add(Task.Factory.StartNew(
                          () =>
                          {
                              dynamicEngine.Start(GetFreshProgram(true, false));
                          },
                          tokenSource.Token));
                    }
                    else
                    {
                        SMTEngine smtEngine = (SMTEngine)engine;
                        underApproximatingTasks.Add(Task.Factory.StartNew(
                          () =>
                          {
                              smtEngine.Start(GetFreshProgram(true, true), ref outcome);
                          },
                          tokenSource.Token));
                    }
                }
            }

            if (pipeline.RunHoudini)
            {
                // We set a barrier on the under-approximating engines if a Houdini delay
                // is specified or no sliding is selected
                if (pipeline.GetHoudiniEngine().Delay > 0)
                {
                    Print.VerboseMessage("Waiting at barrier until Houdini delay has elapsed or all under-approximating engines have finished");
                    Task.WaitAll(underApproximatingTasks.ToArray(), pipeline.GetHoudiniEngine().Delay * 1000);
                }
                else if (pipeline.GetHoudiniEngine().SlidingSeconds > 0)
                {
                    Print.VerboseMessage("Waiting at barrier until all under-approximating engines have finished");
                    Task.WaitAll(underApproximatingTasks.ToArray());
                }

                // Schedule the vanilla Houdini engine
                overApproximatingTasks.Add(Task.Factory.StartNew(
                  () =>
                  {
                      pipeline.GetHoudiniEngine().Start(GetFreshProgram(true, true), ref outcome);
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
                        Thread.Sleep(pipeline.GetHoudiniEngine().SlidingSeconds * 1000);

                        // Only launch a fresh Houdini if the candidate invariant set has changed
                        if (Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count > numOfRefuted)
                        {
                            numOfRefuted = Houdini.ConcurrentHoudini.RefutedSharedAnnotations.Count;

                            VanillaHoudini newHoudiniEngine = new VanillaHoudini(
                                pipeline.GetNextSMTEngineID(),
                                pipeline.GetHoudiniEngine().Solver,
                                pipeline.GetHoudiniEngine().ErrorLimit);
                            pipeline.AddEngine(newHoudiniEngine);

                            Print.VerboseMessage("Scheduling another Houdini instance");

                            overApproximatingTasks.Add(Task.Factory.StartNew(
                              () =>
                              {
                                  newHoudiniEngine.Start(GetFreshProgram(true, true), ref outcome);
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
                    }
                    while (newHoudinis < pipeline.GetHoudiniEngine().SlidingLimit && runningHoudinis);
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

            return outcome;
        }

        private Program GetFreshProgram(bool disableChecks, bool inline)
        {
            return Utilities.GetFreshProgram(this.fileNames, disableChecks, inline);
        }

        private Program ApplyInvariants(Houdini.HoudiniOutcome outcome)
        {
            Program program = GetFreshProgram(false, false);
            CommandLineOptions.Clo.PrintUnstructured = 2;
            Houdini.Houdini.ApplyAssignment(program, outcome);
            if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ReplaceLoopInvariantAssertions)
            {
                foreach (Block block in program.Blocks())
                {
                    List<Cmd> newCmds = new List<Cmd>();
                    foreach (Cmd cmd in block.Cmds)
                    {
                        AssertCmd assertion = cmd as AssertCmd;
                        if (assertion != null
                            && QKeyValue.FindBoolAttribute(assertion.Attributes, "originated_from_invariant"))
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

        private void DumpKilledInvariants(string engineName)
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
