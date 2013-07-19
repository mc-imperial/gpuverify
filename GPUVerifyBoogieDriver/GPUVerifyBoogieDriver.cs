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

        int exitCode = VerifyFiles(fileList);
        Environment.Exit(exitCode);
      } catch (Exception e) {
        if(((GPUVerifyBoogieDriverCommandLineOptions)CommandLineOptions.Clo).DebugGPUVerify) {
          Console.Error.WriteLine("Exception thrown in GPUVerifyBoogieDriver");
          Console.Error.WriteLine(e);
          throw e;
        }

        const string DUMP_FILE = "__gvdump.txt";

        #region Give generic internal error messsage
        Console.Error.WriteLine("\nGPUVerify: an internal error has occurred, details written to " + DUMP_FILE + ".");
        Console.Error.WriteLine();
        Console.Error.WriteLine("Please consult the troubleshooting guide in the GPUVerify documentation");
        Console.Error.WriteLine("for common problems, and if this does not help, raise an issue via the");
        Console.Error.WriteLine("GPUVerify issue tracker:");
        Console.Error.WriteLine();
        Console.Error.WriteLine("  https://gpuverify.codeplex.com");
        Console.Error.WriteLine();
        #endregion"

        #region Now try to give the user a specific hint if this looks like a common problem
        try {
          throw e;
        } catch(ProverException) {
          Console.Error.WriteLine("Hint: It looks like GPUVerify is having trouble invoking its");
          Console.Error.WriteLine("supporting theorem prover, which by default is Z3.  Have you");
          Console.Error.WriteLine("installed Z3?");
        } catch(Exception) {
          // Nothing to say about this
        }
        #endregion

        #region Write details of the exception to the dump file
        using (TokenTextWriter writer = new TokenTextWriter(DUMP_FILE)) {
          writer.Write("Exception ToString:");
          writer.Write("===================");
          writer.Write(e.ToString());
          writer.Close();
        }
        #endregion

        Environment.Exit(1);
      }
    }

    static int VerifyFiles(List<string> fileNames) {
      Contract.Requires(cce.NonNullElements(fileNames));

      Houdini.Houdini houdini = null;

      if (CommandLineOptions.Clo.ContractInfer) {

        #region Compute invariant without race checking
        {
          if (CommandLineOptions.Clo.Trace) {
            Console.WriteLine("Compute invariant without race checking");
          }

          Program InvariantComputationProgram = ParseBoogieProgram(fileNames, false);
          if (InvariantComputationProgram == null) {
            return 1;
          }
          PipelineOutcome oc = ResolveAndTypecheck(InvariantComputationProgram, fileNames[fileNames.Count - 1]);
          if (oc != PipelineOutcome.ResolvedAndTypeChecked)
            return 1;
          DisableRaceChecking(InvariantComputationProgram);
          if (GetCommandLineOptions().ArrayToCheck != null) {
            RestrictToArray(InvariantComputationProgram, GetCommandLineOptions().ArrayToCheck);
          }
          EliminateDeadVariablesAndInline(InvariantComputationProgram);

          var houdiniStats = new Houdini.HoudiniSession.HoudiniStatistics();
          houdini = new Houdini.Houdini(InvariantComputationProgram, houdiniStats);
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
            Console.WriteLine("Prover time = " + houdiniStats.proverTime.ToString("F2"));
            Console.WriteLine("Unsat core prover time = " + houdiniStats.unsatCoreProverTime.ToString("F2"));
            Console.WriteLine("Number of prover queries = " + houdiniStats.numProverQueries);
            Console.WriteLine("Number of unsat core prover queries = " + houdiniStats.numUnsatCoreProverQueries);
            Console.WriteLine("Number of unsat core prunings = " + houdiniStats.numUnsatCorePrunings);
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

        }
        #endregion
      }

      #region Use computed invariant (if any) to perform race checking
      {

        Program RaceCheckingProgram = ParseBoogieProgram(fileNames, false);
        if (RaceCheckingProgram == null) {
          return 1;
        }
        PipelineOutcome oc = ResolveAndTypecheck(RaceCheckingProgram, fileNames[fileNames.Count - 1]);
        if (oc != PipelineOutcome.ResolvedAndTypeChecked)
          return 1;

        if (GetCommandLineOptions().ArrayToCheck != null) {
          RestrictToArray(RaceCheckingProgram, GetCommandLineOptions().ArrayToCheck);
        }
        
        EliminateDeadVariablesAndInline(RaceCheckingProgram);

        CommandLineOptions.Clo.PrintUnstructured = 2;

        if (CommandLineOptions.Clo.LoopUnrollCount != -1) {
          Debug.Assert(!CommandLineOptions.Clo.ContractInfer);
          RaceCheckingProgram.UnrollLoops(CommandLineOptions.Clo.LoopUnrollCount, CommandLineOptions.Clo.SoundLoopUnrolling);
          GPUVerifyErrorReporter.FixStateIds(RaceCheckingProgram);
        }

        if(houdini != null) {
          houdini.ApplyAssignment(RaceCheckingProgram);
        }

        return VerifyProgram(RaceCheckingProgram);
      }
      #endregion

    }

    private static GPUVerifyBoogieDriverCommandLineOptions GetCommandLineOptions() {
      return (GPUVerifyBoogieDriverCommandLineOptions)CommandLineOptions.Clo;
    }

    private static void RestrictToArray(Program prog, string arrayName) {

      if(!ValidArray(prog, arrayName)) {
        ErrorWriteLine("GPUVerify: error: array " + GetCommandLineOptions().ToExternalArrayName(arrayName) + " does not exist");
        Environment.Exit(1);
      }

      var Candidates = prog.TopLevelDeclarations.OfType<Constant>().Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "existential")).Select(Item => Item.Name);

      HashSet<string> CandidatesToRemove = new HashSet<string>();
      foreach (var b in prog.Blocks()) {
        CmdSeq newCmds = new CmdSeq();
        foreach(Cmd c in b.Cmds) {
          var callCmd = c as CallCmd;
          if(callCmd != null && IsRaceInstrumentationProcedureForOtherArray(callCmd, arrayName)) {
            continue;
          }
          var assertCmd = c as AssertCmd;
          if (assertCmd != null && ContainsAccessHasOccurredForOtherArray(assertCmd.Expr, arrayName)) {
            string CandidateName;
            if(Houdini.Houdini.MatchCandidate(assertCmd.Expr, Candidates, out CandidateName)) {
              CandidatesToRemove.Add(CandidateName);
            }
            continue;
          }
          newCmds.Add(c);
        }
        b.Cmds = newCmds;
      }

      foreach (var p in prog.TopLevelDeclarations.OfType<Procedure>()) {
        RequiresSeq newRequires = new RequiresSeq();
        foreach (Requires r in p.Requires) {
          if (ContainsAccessHasOccurredForOtherArray(r.Condition, arrayName)) {
            continue;
          }
          newRequires.Add(r);
        }
        p.Requires = newRequires;

        EnsuresSeq newEnsures = new EnsuresSeq();
        foreach (Ensures r in p.Ensures) {
          if (ContainsAccessHasOccurredForOtherArray(r.Condition, arrayName)) {
            continue;
          }
          newEnsures.Add(r);
        }
        p.Ensures = newEnsures;
      }

      prog.TopLevelDeclarations.RemoveAll(Item => (Item is Variable) &&
        CandidatesToRemove.Contains((Item as Variable).Name));

    }

    private static bool ValidArray(Program prog, string arrayName) {
      return prog.TopLevelDeclarations.OfType<Variable>().Where(Item =>
        QKeyValue.FindBoolAttribute(Item.Attributes, "race_checking") &&
        Item.Name.StartsWith("_WRITE_HAS_OCCURRED_")).Select(Item => Item.Name).Contains("_WRITE_HAS_OCCURRED_" + arrayName + "$1");
    }

    class FindAccessHasOccurredForOtherArrayVisitor : StandardVisitor {

      private string arrayName;
      private bool Found;

      internal FindAccessHasOccurredForOtherArrayVisitor(string arrayName) {
        this.arrayName = arrayName;
        this.Found = false;
      }

      public override Variable VisitVariable(Variable node) {
        foreach (var AccessType in new string[] { "READ", "WRITE" }) {
          string prefix = "_" + AccessType + "_HAS_OCCURRED_";
          if (node.Name.StartsWith(prefix)) {
            if (!node.Name.Substring(prefix.Length).Equals(arrayName + "$1")) {
              Found = true;
              return node;
            }
          }
        }
        return node;
      }

      internal bool IsFound() {
        return Found;
      }

    }

    private static bool ContainsAccessHasOccurredForOtherArray(Expr expr, string arrayName) {
      var v = new FindAccessHasOccurredForOtherArrayVisitor(arrayName);
      v.VisitExpr(expr);
      return v.IsFound();
    }

    private static bool IsRaceInstrumentationProcedureForOtherArray(CallCmd callCmd, string arrayName) {
      foreach (var AccessType in new string[] { "READ", "WRITE" }) {
        foreach (var ProcedureType in new string[] { "LOG", "CHECK" }) {
          var prefix = "_" + ProcedureType + "_" + AccessType + "_";
          if(callCmd.callee.StartsWith(prefix)) {
            return !callCmd.callee.Substring(prefix.Length).Equals(arrayName);
          }
        }
      }
      return false;
    }


    private static int VerifyProgram(Program program) {
      int errorCount = 0;
      int verified = 0;
      int inconclusives = 0;
      int timeOuts = 0;
      int outOfMemories = 0;

      ConditionGeneration vcgen = null;
      try {
        vcgen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend, new List<Checker>());
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
      foreach (var block in program.Blocks()) {
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
      foreach (var block in program.Blocks()) {
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

      LinearTypechecker linearTypechecker = new LinearTypechecker(program);
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
