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
  using System.Collections.Generic;
  using System.Diagnostics.Contracts;
  using System.Diagnostics;
  using System.Linq;
  using VC;

  public class GPUVerifyBoogieDriver
  {
    public static void Main(string[] args)
    {
      Contract.Requires(cce.NonNullElements(args));
      CommandLineOptions.Install(new GPUVerifyKernelAnalyserCommandLineOptions());

      try {

        CommandLineOptions.Clo.RunningBoogieFromCommandLine = true;
        if (!CommandLineOptions.Clo.Parse(args)) {
          Environment.Exit(1);
        }

        if (CommandLineOptions.Clo.Files.Count == 0) {
          GVUtil.IO.ErrorWriteLine("GPUVerify: error: no input files were specified");
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
            GVUtil.IO.ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
            Environment.Exit(1);
          }
        }

        int exitCode = VerifyFiles(fileList);
        Environment.Exit(exitCode);
      } catch (Exception e) {
        if(GetCommandLineOptions().DebugGPUVerify) {
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

    static int VerifyFiles(List<string> fileNames)
    {
      Contract.Requires(cce.NonNullElements(fileNames));

      Program program = GVUtil.IO.ParseBoogieProgram(fileNames, false);
      if (program == null) return 1;

      KernelAnalyser.PipelineOutcome oc = KernelAnalyser.ResolveAndTypecheck(program, fileNames[fileNames.Count - 1]);
      if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked) return 1;

      KernelAnalyser.EliminateDeadVariablesAndInline(program);
      KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(program);

      CommandLineOptions.Clo.PrintUnstructured = 2;

      if (CommandLineOptions.Clo.LoopUnrollCount != -1) {
        Debug.Assert(!CommandLineOptions.Clo.ContractInfer);
        program.UnrollLoops(CommandLineOptions.Clo.LoopUnrollCount, CommandLineOptions.Clo.SoundLoopUnrolling);
        GPUVerifyErrorReporter.FixStateIds(program);
      }

      return VerifyProgram(program);
    }

    private static int VerifyProgram(Program program)
    {
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
        GVUtil.IO.ErrorWriteLine("Fatal Error: ProverException: {0}", e);
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
            GVUtil.IO.ReportBplError(impl, String.Format("Error BP5010: {0}  Encountered in implementation {1}.", e.Message, impl.Name), true, true);
            errors = null;
            outcome = VCGen.Outcome.Inconclusive;
          }
          catch (UnexpectedProverOutputException upo) {
            GVUtil.IO.AdvisoryWriteLine("Advisory: {0} SKIPPED because of internal error: unexpected prover output: {1}", impl.Name, upo.Message);
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

          KernelAnalyser.ProcessOutcome(outcome, errors, timeIndication, ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);

          if (outcome == VCGen.Outcome.Errors || CommandLineOptions.Clo.Trace) {
            Console.Out.Flush();
          }
        }
      }

      vcgen.Close();
      cce.NonNull(CommandLineOptions.Clo.TheProverFactory).Close();

      GVUtil.IO.WriteTrailer(verified, errorCount, inconclusives, timeOuts, outOfMemories);

      return errorCount + inconclusives + timeOuts + outOfMemories;
    }

    private static void AddArrayToggles(Program RaceCheckingProgram)
    {
      Dictionary<string, Constant> ToggleVars = new Dictionary<string, Constant>();
      foreach(var p in RaceCheckingProgram.TopLevelDeclarations.OfType<Procedure>()) {
        foreach(Requires r in p.Requires) {
          if(QKeyValue.FindBoolAttribute(r.Attributes, "race")) {
            string arrayName;
            if(p.Name.StartsWith("_CHECK_READ_")) {
              arrayName = p.Name.Substring("_CHECK_READ_".Length);
            } else if(p.Name.StartsWith("_CHECK_WRITE_")) {
              arrayName = p.Name.Substring("_CHECK_WRITE_".Length);
            } else if(p.Name.StartsWith("_CHECK_ATOMIC_")) {
              arrayName = p.Name.Substring("_CHECK_ATOMIC_".Length);
            } else {
              continue;
            }
            if(!ToggleVars.ContainsKey(arrayName)) {
              ToggleVars[arrayName] = new Constant(Token.NoToken, new TypedIdent(Token.NoToken, "__toggle_" + arrayName, Type.Bool), false);
              ToggleVars[arrayName].AddAttribute("toggle", new object[] { Expr.True });
            }
            Constant toggleVar = ToggleVars[arrayName];
            r.Condition = Expr.Imp(new IdentifierExpr(Token.NoToken, toggleVar), r.Condition);
          }
        }
      }
      RaceCheckingProgram.TopLevelDeclarations.AddRange(ToggleVars.Values);
    }

    private static void RestrictToArray(Program prog, string arrayName)
    {
      if(!ValidArray(prog, arrayName)) {
        GVUtil.IO.ErrorWriteLine("GPUVerify: error: array " + GetCommandLineOptions().ToExternalArrayName(arrayName) + " does not exist");
        Environment.Exit(1);
      }

      var Candidates = prog.TopLevelDeclarations.OfType<Constant>().Where(Item => QKeyValue.FindBoolAttribute(Item.Attributes, "existential")).Select(Item => Item.Name);

      HashSet<string> CandidatesToRemove = new HashSet<string>();
      foreach (var b in prog.Blocks()) {
        List<Cmd> newCmds = new List<Cmd>();
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
        List<Requires> newRequires = new List<Requires>();
        foreach (Requires r in p.Requires) {
          if (ContainsAccessHasOccurredForOtherArray(r.Condition, arrayName)) {
            continue;
          }
          newRequires.Add(r);
        }
        p.Requires = newRequires;

        List<Ensures> newEnsures = new List<Ensures>();
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

    private static bool ValidArray(Program prog, string arrayName)
    {
      return prog.TopLevelDeclarations.OfType<Variable>().Where(Item =>
        QKeyValue.FindBoolAttribute(Item.Attributes, "race_checking") &&
        Item.Name.StartsWith("_WRITE_HAS_OCCURRED_")).Select(Item => Item.Name).Contains("_WRITE_HAS_OCCURRED_" + arrayName + "$1");
    }

    class FindAccessHasOccurredForOtherArrayVisitor : StandardVisitor
    {
      private string arrayName;
      private bool Found;

      internal FindAccessHasOccurredForOtherArrayVisitor(string arrayName) {
        this.arrayName = arrayName;
        this.Found = false;
      }

      public override Variable VisitVariable(Variable node) {
        foreach (var Access in AccessType.Types) {
          string prefix = "_" + Access + "_HAS_OCCURRED_";
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

    private static bool ContainsAccessHasOccurredForOtherArray(Expr expr, string arrayName)
    {
      var v = new FindAccessHasOccurredForOtherArrayVisitor(arrayName);
      v.VisitExpr(expr);
      return v.IsFound();
    }

    private static bool IsRaceInstrumentationProcedureForOtherArray(CallCmd callCmd, string arrayName)
    {
      foreach (var Access in AccessType.Types) {
        foreach (var ProcedureType in new string[] { "LOG", "CHECK" }) {
          var prefix = "_" + ProcedureType + "_" + Access + "_";
          if(callCmd.callee.StartsWith(prefix)) {
            return !callCmd.callee.Substring(prefix.Length).Equals(arrayName);
          }
        }
      }
      return false;
    }

    private static GPUVerifyKernelAnalyserCommandLineOptions GetCommandLineOptions()
    {
      return (GPUVerifyKernelAnalyserCommandLineOptions)CommandLineOptions.Clo;
    }
  }
}
