//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using GPUVerify;

namespace Microsoft.Boogie
{
  using System;
  using System.IO;
  using System.Collections;
  using System.Collections.Generic;
  using PureCollections;
  using Microsoft.Boogie;
  using Microsoft.Boogie.AbstractInterpretation;
  using System.Diagnostics.Contracts;
  using System.Diagnostics;
  using System.Linq;
  using VC;
  using BoogiePL = Microsoft.Boogie;
  
  /* 
    The following assemblies are referenced because they are needed at runtime, not at compile time:
      BaseTypes
      Provers.Z3
      System.Compiler.Framework
  */

  public class GPUVerifyBoogieDriver {

    public static void Main(string[] args) {
      Contract.Requires(cce.NonNullElements(args));
      CommandLineOptions.Install(new GPUVerifyBoogieDriverCommandLineOptions());

      try {

        CommandLineOptions.Clo.RunningBoogieFromCommandLine = true;
        if (!CommandLineOptions.Clo.Parse(args)) {
          Environment.Exit(1);
        }

        if (CommandLineOptions.Clo.Files.Count == 0) {
          ErrorWriteLine("GPUVerify: error: no input files were specified");
          Environment.Exit(1);
        }
        if (!CommandLineOptions.Clo.DontShowLogo) {
          Console.WriteLine(CommandLineOptions.Clo.Version);
        }

        List<string> fileList = new List<string>();
        foreach (string file in CommandLineOptions.Clo.Files) {
          string extension = Path.GetExtension(file);
          if (extension != null) {
            extension = extension.ToLower();
          }
          fileList.Add(file);
        }
        foreach (string file in fileList) {
          Contract.Assert(file != null);
          string extension = Path.GetExtension(file);
          if (extension != null) {
            extension = extension.ToLower();
          }
          if (extension != ".bpl") {
            ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
            Environment.Exit(1);
          }
        }

        int exitCode;

        if (CommandLineOptions.Clo.ContractInfer) {
          exitCode = VerifyWithInference(fileList);
        }
        else {
          exitCode = VerifyDirectly(fileList);
        }
        Environment.Exit(exitCode);
      } catch (Exception e) {
        Console.Error.WriteLine("Exception thrown in GPUVerifyBoogieDriver");
        Console.Error.WriteLine(e);
        Environment.Exit(1);
      }
    }

    static int VerifyDirectly(List<string> fileNames) {
      Contract.Requires(cce.NonNullElements(fileNames));
      Contract.Requires(!CommandLineOptions.Clo.ContractInfer);

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("Verifying without inference");
      }

      Program program = ParseBoogieProgram(fileNames, false);
      if (program == null) {
        return 1;
      }

      PipelineOutcome oc = ResolveAndTypecheck(program, fileNames[fileNames.Count - 1]);
      if (oc != PipelineOutcome.ResolvedAndTypeChecked)
        return 1;

      EliminateDeadVariablesAndInline(program);

      if (CommandLineOptions.Clo.LoopUnrollCount != -1) {
        program.UnrollLoops(CommandLineOptions.Clo.LoopUnrollCount, CommandLineOptions.Clo.SoundLoopUnrolling);
      }

      return VerifyProgram(program);

    }

    static int VerifyWithInference(List<string> fileNames) {
      Contract.Requires(cce.NonNullElements(fileNames));
      Contract.Requires(CommandLineOptions.Clo.ContractInfer);
      Contract.Requires(CommandLineOptions.Clo.LoopUnrollCount == -1);

      if (CommandLineOptions.Clo.Trace) {
        Console.WriteLine("Verifying with inference");
      }

      List<Houdini.Houdini> HoudiniInstances = new List<Houdini.Houdini>();

      #region Compute invariant without race checking
      {
        if (CommandLineOptions.Clo.Trace) {
          Console.WriteLine("Compute invariant without race checking");
        }

        int CurrentStage = -1;
        List<int> Stages = null;

        do {
          Program InvariantComputationProgram = ParseBoogieProgram(fileNames, false);
          if (InvariantComputationProgram == null) {
            return 1;
          }
          PipelineOutcome oc = ResolveAndTypecheck(InvariantComputationProgram, fileNames[fileNames.Count - 1]);
          if (oc != PipelineOutcome.ResolvedAndTypeChecked)
            return 1;

          if(Stages == null) {
            Stages = InvariantComputationProgram.TopLevelDeclarations.OfType<Constant>().Where(Item
              => QKeyValue.FindIntAttribute(Item.Attributes, "stage_id", -1) != -1).Select(Item =>
                QKeyValue.FindIntAttribute(Item.Attributes, "stage_id", -1)).ToList();

          }

          if (Stages.Count == 0) {
            break;
          }
            
          do {
            CurrentStage++;
          } while(!Stages.Contains(CurrentStage));

          if (CommandLineOptions.Clo.Trace) {
            Console.WriteLine("Current inference stage: " + CurrentStage);
          }

          if((CommandLineOptions.Clo as GPUVerifyBoogieDriverCommandLineOptions).StagedInference && 
                CurrentStage == InferenceStages.BASIC_CANDIDATE_STAGE) {
            DisableRaceLogging(InvariantComputationProgram);
          }

          DisableRaceChecking(InvariantComputationProgram);

          EliminateDeadVariablesAndInline(InvariantComputationProgram);

          // Instantiate or remove candidates based on what has been
          // learned during previous iterations
          foreach(var h in HoudiniInstances) {
            h.ApplyAssignment(InvariantComputationProgram, true);
          }

          DisableCandidatesFromHigherStages(InvariantComputationProgram, CurrentStage);

          Houdini.Houdini houdini = new Houdini.Houdini(InvariantComputationProgram);
          HoudiniInstances.Add(houdini);

          Houdini.HoudiniOutcome outcome = houdini.PerformHoudiniInference();
          if (CommandLineOptions.Clo.PrintAssignment) {
            Console.WriteLine("Assignment computed by Houdini:");
            foreach (var x in outcome.assignment) {
              Console.WriteLine(x.Key + " = " + x.Value);
            }
          }
          if (CommandLineOptions.Clo.Trace) {
            int numTrueAssigns = 0;
            foreach (var x in outcome.assignment) {
              if (x.Value)
                numTrueAssigns++;
            }
            Console.WriteLine("Number of true assignments = " + numTrueAssigns);
            Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));
            Console.WriteLine("Prover time = " + Houdini.HoudiniSession.proverTime.ToString("F2"));
            Console.WriteLine("Unsat core prover time = " + Houdini.HoudiniSession.unsatCoreProverTime.ToString("F2"));
            Console.WriteLine("Number of prover queries = " + Houdini.HoudiniSession.numProverQueries);
            Console.WriteLine("Number of unsat core prover queries = " + Houdini.HoudiniSession.numUnsatCoreProverQueries);
            Console.WriteLine("Number of unsat core prunings = " + Houdini.HoudiniSession.numUnsatCorePrunings);
          }

          if (!AllImplementationsValid(outcome)) {
            int verified = 0;
            int errorCount = 0;
            int inconclusives = 0;
            int timeOuts = 0;
            int outOfMemories = 0;
            foreach (Houdini.VCGenOutcome x in outcome.implementationOutcomes.Values) {
              ProcessOutcome(x.outcome, x.errors, "", ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);
            }
            WriteTrailer(verified, errorCount, inconclusives, timeOuts, outOfMemories);
            return errorCount + inconclusives + timeOuts + outOfMemories;
          }

          Console.WriteLine("Max stages is " + Stages.Max());

        } while(CurrentStage < Stages.Max());
      }
      #endregion

      #region Use computed invariant to perform race checking
      {
        if (CommandLineOptions.Clo.Trace) {
          Console.WriteLine("Use computed invariant to perform race checking");
        }

        Program RaceCheckingProgram = ParseBoogieProgram(fileNames, false);
        if (RaceCheckingProgram == null) {
          return 1;
        }
        PipelineOutcome oc = ResolveAndTypecheck(RaceCheckingProgram, fileNames[fileNames.Count - 1]);
        if (oc != PipelineOutcome.ResolvedAndTypeChecked)
          return 1;
        EliminateDeadVariablesAndInline(RaceCheckingProgram);

        foreach (var h in HoudiniInstances) {
          h.ApplyAssignment(RaceCheckingProgram, true);
        }

        CommandLineOptions.Clo.PrintUnstructured = 2;

        return VerifyProgram(RaceCheckingProgram);
      }
      #endregion

    }

    private static void DisableCandidatesFromHigherStages(Program program, int CurrentStage) {
      Contract.Requires(CurrentStage >= 0);
      var CandidatesToDisable =
        program.TopLevelDeclarations.OfType<Constant>().
          Where(Item => QKeyValue.FindIntAttribute(Item.Attributes, "stage_id", -1) > CurrentStage).
            Select(Item => Item.Name);

      // Treat all assertions
      // TODO: do we need to also consider assumptions?
      foreach (Block block in program.TopLevelDeclarations.OfType<Implementation>().Select(item => item.Blocks).SelectMany(item => item)) {
        CmdSeq newCmds = new CmdSeq();
        foreach (Cmd cmd in block.Cmds) {
          string c;
          AssertCmd assertCmd = cmd as AssertCmd;
          if (assertCmd == null || !Houdini.Houdini.MatchCandidate(assertCmd.Expr, CandidatesToDisable, out c)) {
            newCmds.Add(cmd);
          }
        }
        block.Cmds = newCmds;
      }

      // Treat requires and ensures
      new DisableCandidatesVisitor(CandidatesToDisable, program).VisitProgram(program);

      // Remove the existential constants
      program.TopLevelDeclarations.RemoveAll(item => (item is Variable) &&
           (CandidatesToDisable.Contains((item as Variable).Name)));

    }

    class DisableCandidatesVisitor : StandardVisitor {

      private IEnumerable<string> CandidatesToDisable;
      private Program prog;

      internal DisableCandidatesVisitor(IEnumerable<string> CandidatesToDisable, Program prog) {
        this.CandidatesToDisable = CandidatesToDisable;
        this.prog = prog;
      }

      public override Requires VisitRequires(Requires requires) {
        requires.Condition = DisableCandidates(requires.Condition);
        return requires;
      }

      public override Ensures VisitEnsures(Ensures ensures) {
        ensures.Condition = DisableCandidates(ensures.Condition);
        return ensures;
      }

      private Expr DisableCandidates(Expr e) {
        string c;
        if (Houdini.Houdini.MatchCandidate(e, CandidatesToDisable, out c)) {
          return Expr.True;
        }
        return e;
      }

    }



    private static int VerifyProgram(Program program) {
      int errorCount = 0;
      int verified = 0;
      int inconclusives = 0;
      int timeOuts = 0;
      int outOfMemories = 0;

      ConditionGeneration vcgen = null;
      try {
        vcgen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend);
      }
      catch (ProverException e) {
        ErrorWriteLine("Fatal Error: ProverException: {0}", e);
        return 1;
      }

      // operate on a stable copy, in case it gets updated while we're running
      var decls = program.TopLevelDeclarations.ToArray();
      foreach (Declaration decl in decls) {
        Contract.Assert(decl != null);
        int prevAssertionCount = vcgen.CumulativeAssertionCount;
        Implementation impl = decl as Implementation;
        if (impl != null && CommandLineOptions.Clo.UserWantsToCheckRoutine(cce.NonNull(impl.Name)) && !impl.SkipVerification) {
          List<Counterexample/*!*/>/*?*/ errors;

          DateTime start = new DateTime();  // to please compiler's definite assignment rules
          if (CommandLineOptions.Clo.Trace) {
            start = DateTime.UtcNow;
            if (CommandLineOptions.Clo.Trace) {
              Console.WriteLine();
              Console.WriteLine("Verifying {0} ...", impl.Name);
            }
          }

          VCGen.Outcome outcome;
          try {
            outcome = vcgen.VerifyImplementation(impl, out errors);
          }
          catch (VCGenException e) {
            ReportBplError(impl, String.Format("Error BP5010: {0}  Encountered in implementation {1}.", e.Message, impl.Name), true, true);
            errors = null;
            outcome = VCGen.Outcome.Inconclusive;
          }
          catch (UnexpectedProverOutputException upo) {
            AdvisoryWriteLine("Advisory: {0} SKIPPED because of internal error: unexpected prover output: {1}", impl.Name, upo.Message);
            errors = null;
            outcome = VCGen.Outcome.Inconclusive;
          }

          string timeIndication = "";
          DateTime end = DateTime.UtcNow;
          TimeSpan elapsed = end - start;
          if (CommandLineOptions.Clo.Trace) {
            int poCount = vcgen.CumulativeAssertionCount - prevAssertionCount;
            timeIndication = string.Format("  [{0:F3} s, {1} proof obligation{2}]  ", elapsed.TotalSeconds, poCount, poCount == 1 ? "" : "s");
          }

          ProcessOutcome(outcome, errors, timeIndication, ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);

          if (outcome == VCGen.Outcome.Errors || CommandLineOptions.Clo.Trace) {
            Console.Out.Flush();
          }
        }
      }

      vcgen.Close();
      cce.NonNull(CommandLineOptions.Clo.TheProverFactory).Close();

      WriteTrailer(verified, errorCount, inconclusives, timeOuts, outOfMemories);

      return errorCount + inconclusives + timeOuts + outOfMemories;
    }


    private static void DisableRaceChecking(Program program) {
      foreach (var block in program.TopLevelDeclarations.OfType<Implementation>().Select(item => item.Blocks).SelectMany(item => item)) {
        CmdSeq newCmds = new CmdSeq();
        foreach (Cmd c in block.Cmds) {
          CallCmd callCmd = c as CallCmd;
          // TODO: refine into proper check
          if(callCmd == null || !(callCmd.callee.Contains("_CHECK_READ") || 
                                  callCmd.callee.Contains("_CHECK_WRITE"))) {
            newCmds.Add(c);
          }
        }
        block.Cmds = newCmds;
      }
    }

    private static void DisableRaceLogging(Program program) {
      foreach (var block in program.TopLevelDeclarations.OfType<Implementation>().Select(item => item.Blocks).SelectMany(item => item)) {
        CmdSeq newCmds = new CmdSeq();
        foreach (Cmd c in block.Cmds) {
          CallCmd callCmd = c as CallCmd;
          // TODO: refine into proper check
          if (callCmd == null || !(callCmd.callee.Contains("_LOG_READ") ||
                                  callCmd.callee.Contains("_LOG_WRITE"))) {
            newCmds.Add(c);
          }
        }
        block.Cmds = newCmds;
      }
    }


    public static void ErrorWriteLine(string s) {
      Contract.Requires(s != null);
      ConsoleColor col = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.DarkGray;
      Console.Error.WriteLine(s);
      Console.ForegroundColor = col;
    }

    public static void ErrorWriteLine(string format, params object[] args) {
      Contract.Requires(format != null);
      string s = string.Format(format, args);
      ErrorWriteLine(s);
    }

    public static void AdvisoryWriteLine(string format, params object[] args) {
      Contract.Requires(format != null);
      ConsoleColor col = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.Yellow;
      Console.WriteLine(format, args);
      Console.ForegroundColor = col;
    }





    /// <summary>
    /// Inform the user about something and proceed with translation normally.
    /// Print newline after the message.
    /// </summary>
    public static void Inform(string s) {
      if (CommandLineOptions.Clo.Trace || CommandLineOptions.Clo.TraceProofObligations) {
        Console.WriteLine(s);
      }
    }

    static void WriteTrailer(int verified, int errors, int inconclusives, int timeOuts, int outOfMemories) {
      Contract.Requires(0 <= errors && 0 <= inconclusives && 0 <= timeOuts && 0 <= outOfMemories);
      Console.WriteLine();
      if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
        Console.Write("{0} finished with {1} credible, {2} doomed{3}", CommandLineOptions.Clo.DescriptiveToolName, verified, errors, errors == 1 ? "" : "s");
      } else {
        Console.Write("{0} finished with {1} verified, {2} error{3}", CommandLineOptions.Clo.DescriptiveToolName, verified, errors, errors == 1 ? "" : "s");
      }
      if (inconclusives != 0) {
        Console.Write(", {0} inconclusive{1}", inconclusives, inconclusives == 1 ? "" : "s");
      }
      if (timeOuts != 0) {
        Console.Write(", {0} time out{1}", timeOuts, timeOuts == 1 ? "" : "s");
      }
      if (outOfMemories != 0) {
        Console.Write(", {0} out of memory", outOfMemories);
      }
      Console.WriteLine();
      Console.Out.Flush();
    }



    static void ReportBplError(Absy node, string message, bool error, bool showBplLocation) {
      Contract.Requires(message != null);
      Contract.Requires(node != null);
      IToken tok = node.tok;
      string s;
      if (tok != null && showBplLocation) {
        s = string.Format("{0}({1},{2}): {3}", tok.filename, tok.line, tok.col, message);
      } else {
        s = message;
      }
      if (error) {
        ErrorWriteLine(s);
      } else {
        Console.WriteLine(s);
      }
    }


    static void ProcessOutcome(VC.VCGen.Outcome outcome, List<Counterexample> errors, string timeIndication,
                       ref int errorCount, ref int verified, ref int inconclusives, ref int timeOuts, ref int outOfMemories) {

      switch (outcome) {
        default:
          Contract.Assert(false);  // unexpected outcome
          throw new cce.UnreachableException();
        case VCGen.Outcome.ReachedBound:
          Inform(String.Format("{0}verified", timeIndication));
          Console.WriteLine(string.Format("Stratified Inlining: Reached recursion bound of {0}", CommandLineOptions.Clo.RecursionBound));
          verified++;
          break;
        case VCGen.Outcome.Correct:
          if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
            Inform(String.Format("{0}credible", timeIndication));
            verified++;
          }
          else {
            Inform(String.Format("{0}verified", timeIndication));
            verified++;
          }
          break;
        case VCGen.Outcome.TimedOut:
          timeOuts++;
          Inform(String.Format("{0}timed out", timeIndication));
          break;
        case VCGen.Outcome.OutOfMemory:
          outOfMemories++;
          Inform(String.Format("{0}out of memory", timeIndication));
          break;
        case VCGen.Outcome.Inconclusive:
          inconclusives++;
          Inform(String.Format("{0}inconclusive", timeIndication));
          break;
        case VCGen.Outcome.Errors:
          if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
            Inform(String.Format("{0}doomed", timeIndication));
            errorCount++;
          } //else {
          Contract.Assert(errors != null);  // guaranteed by postcondition of VerifyImplementation
          {
            // BP1xxx: Parsing errors
            // BP2xxx: Name resolution errors
            // BP3xxx: Typechecking errors
            // BP4xxx: Abstract interpretation errors (Is there such a thing?)
            // BP5xxx: Verification errors

            errors.Sort(new CounterexampleComparer());
            foreach (Counterexample error in errors)
            {
              GPUVerifyErrorReporter.ReportCounterexample(error);
              errorCount++;
            }
            //}
            Inform(String.Format("{0}error{1}", timeIndication, errors.Count == 1 ? "" : "s"));
          }
          break;
      }
    }


    private static bool AllImplementationsValid(Houdini.HoudiniOutcome outcome) {
      foreach (var vcgenOutcome in outcome.implementationOutcomes.Values.Select(i => i.outcome)) {
        if (vcgenOutcome != VCGen.Outcome.Correct) {
          return false;
        }
      }
      return true;
    }









    // To go to library


    /// <summary>
    /// Parse the given files into one Boogie program.  If an I/O or parse error occurs, an error will be printed
    /// and null will be returned.  On success, a non-null program is returned.
    /// </summary>
    static Program ParseBoogieProgram(List<string> fileNames, bool suppressTraceOutput) {
      Contract.Requires(cce.NonNullElements(fileNames));
      //BoogiePL.Errors.count = 0;
      Program program = null;
      bool okay = true;
      for (int fileId = 0; fileId < fileNames.Count; fileId++) {
        string bplFileName = fileNames[fileId];
        if (!suppressTraceOutput) {
          if (CommandLineOptions.Clo.XmlSink != null) {
            CommandLineOptions.Clo.XmlSink.WriteFileFragment(bplFileName);
          }
          if (CommandLineOptions.Clo.Trace) {
            Console.WriteLine("Parsing " + bplFileName);
          }
        }

        Program programSnippet;
        int errorCount;
        try {
          var defines = new List<string>() { "FILE_" + fileId };
          errorCount = BoogiePL.Parser.Parse(bplFileName, defines, out programSnippet);
          if (programSnippet == null || errorCount != 0) {
            Console.WriteLine("{0} parse errors detected in {1}", errorCount, bplFileName);
            okay = false;
            continue;
          }
        }
        catch (IOException e) {
          ErrorWriteLine("Error opening file \"{0}\": {1}", bplFileName, e.Message);
          okay = false;
          continue;
        }
        if (program == null) {
          program = programSnippet;
        }
        else if (programSnippet != null) {
          program.TopLevelDeclarations.AddRange(programSnippet.TopLevelDeclarations);
        }
      }
      if (!okay) {
        return null;
      }
      else if (program == null) {
        return new Program();
      }
      else {
        return program;
      }
    }


    enum PipelineOutcome {
      Done,
      ResolutionError,
      TypeCheckingError,
      ResolvedAndTypeChecked,
      FatalError,
      VerificationCompleted
    }

    /// <summary>
    /// Resolves and type checks the given Boogie program.  Any errors are reported to the
    /// console.  Returns:
    ///  - Done if no errors occurred, and command line specified no resolution or no type checking.
    ///  - ResolutionError if a resolution error occurred
    ///  - TypeCheckingError if a type checking error occurred
    ///  - ResolvedAndTypeChecked if both resolution and type checking succeeded
    /// </summary>
    static PipelineOutcome ResolveAndTypecheck(Program program, string bplFileName) {
      Contract.Requires(program != null);
      Contract.Requires(bplFileName != null);
      // ---------- Resolve ------------------------------------------------------------

      if (CommandLineOptions.Clo.NoResolve) {
        return PipelineOutcome.Done;
      }

      int errorCount = program.Resolve();
      if (errorCount != 0) {
        Console.WriteLine("{0} name resolution errors detected in {1}", errorCount, bplFileName);
        return PipelineOutcome.ResolutionError;
      }

      // ---------- Type check ------------------------------------------------------------

      if (CommandLineOptions.Clo.NoTypecheck) {
        return PipelineOutcome.Done;
      }

      errorCount = program.Typecheck();
      if (errorCount != 0) {
        Console.WriteLine("{0} type checking errors detected in {1}", errorCount, bplFileName);
        return PipelineOutcome.TypeCheckingError;
      }

      LinearTypechecker linearTypechecker = new LinearTypechecker();
      linearTypechecker.VisitProgram(program);
      if (linearTypechecker.errorCount > 0) {
        Console.WriteLine("{0} type checking errors detected in {1}", errorCount, bplFileName);
        return PipelineOutcome.TypeCheckingError;
      }

      if (CommandLineOptions.Clo.PrintFile != null && CommandLineOptions.Clo.PrintDesugarings) {
        // if PrintDesugaring option is engaged, print the file here, after resolution and type checking
        PrintBplFile(CommandLineOptions.Clo.PrintFile, program, true);
      }

      return PipelineOutcome.ResolvedAndTypeChecked;
    }

    static void EliminateDeadVariablesAndInline(Program program) {
      Contract.Requires(program != null);
      // Eliminate dead variables
      Microsoft.Boogie.UnusedVarEliminator.Eliminate(program);

      // Collect mod sets
      if (CommandLineOptions.Clo.DoModSetAnalysis) {
        Microsoft.Boogie.ModSetCollector.DoModSetAnalysis(program);
      }

      // Coalesce blocks
      if (CommandLineOptions.Clo.CoalesceBlocks) {
        if (CommandLineOptions.Clo.Trace)
          Console.WriteLine("Coalescing blocks...");
        Microsoft.Boogie.BlockCoalescer.CoalesceBlocks(program);
      }

      // Inline
      var TopLevelDeclarations = cce.NonNull(program.TopLevelDeclarations);

      if (CommandLineOptions.Clo.ProcedureInlining != CommandLineOptions.Inlining.None) {
        bool inline = false;
        foreach (var d in TopLevelDeclarations) {
          if (d.FindExprAttribute("inline") != null) {
            inline = true;
          }
        }
        if (inline) {
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null) {
              impl.OriginalBlocks = impl.Blocks;
              impl.OriginalLocVars = impl.LocVars;
            }
          }
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null && !impl.SkipVerification) {
              Inliner.ProcessImplementation(program, impl);
            }
          }
          foreach (var d in TopLevelDeclarations) {
            var impl = d as Implementation;
            if (impl != null) {
              impl.OriginalBlocks = null;
              impl.OriginalLocVars = null;
            }
          }
        }
      }
    }

    static void PrintBplFile(string filename, Program program, bool allowPrintDesugaring) {
      Contract.Requires(program != null);
      Contract.Requires(filename != null);
      bool oldPrintDesugaring = CommandLineOptions.Clo.PrintDesugarings;
      if (!allowPrintDesugaring) {
        CommandLineOptions.Clo.PrintDesugarings = false;
      }
      using (TokenTextWriter writer = filename == "-" ?
                                      new TokenTextWriter("<console>", Console.Out) :
                                      new TokenTextWriter(filename)) {
        if (CommandLineOptions.Clo.ShowEnv != CommandLineOptions.ShowEnvironment.Never) {
          writer.WriteLine("// " + CommandLineOptions.Clo.Version);
          writer.WriteLine("// " + CommandLineOptions.Clo.Environment);
        }
        writer.WriteLine();
        program.Emit(writer);
      }
      CommandLineOptions.Clo.PrintDesugarings = oldPrintDesugaring;
    }


  }

}
