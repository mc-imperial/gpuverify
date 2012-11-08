using GPUVerifyBoogieDriver;
//-----------------------------------------------------------------------------
//
// Copyright (C) Microsoft Corporation.  All Rights Reserved.
//
//-----------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------
// OnlyBoogie OnlyBoogie.ssc
//       - main program for taking a BPL program and verifying it
//---------------------------------------------------------------------------------------------

namespace Microsoft.Boogie
{
  using System;
  using System.Collections;
  using System.Collections.Generic;
  using System.IO;
  using System.Text.RegularExpressions;
  using PureCollections;
  using Microsoft.Basetypes;
  using Microsoft.Boogie;
  using Microsoft.Boogie.AbstractInterpretation;
  using System.Diagnostics.Contracts;
  using System.Diagnostics;
  using System.Linq;
  using VC;
  using AI = Microsoft.Boogie.AbstractInterpretation;
  using BoogiePL = Microsoft.Boogie;
  
  /* 
    The following assemblies are referenced because they are needed at runtime, not at compile time:
      BaseTypes
      Provers.Z3
      System.Compiler.Framework
  */

  public class GPUVerifyBoogieDriver
  {
    // ------------------------------------------------------------------------
    // Main

    public static void Main(string[] args) {
      Contract.Requires(cce.NonNullElements(args));
      CommandLineOptions.Install(new GPUVerifyBoogieDriverCommandLineOptions());

      try {

        int exitCode;

        CommandLineOptions.Clo.RunningBoogieFromCommandLine = true; /* NEEDED? */
        if (!CommandLineOptions.Clo.Parse(args)) {
          Environment.Exit(1);
        }

        if (CommandLineOptions.Clo.Files.Count == 0) {
          ErrorWriteLine("GPUVerify: error: no input files were specified");
          Environment.Exit(1);
        }
        if (CommandLineOptions.Clo.XmlSink != null) {
          string errMsg = CommandLineOptions.Clo.XmlSink.Open();
          if (errMsg != null) {
            ErrorWriteLine("GPUVerify: error: " + errMsg);
            exitCode = 1;
            goto END;
          }
        }
        if (!CommandLineOptions.Clo.DontShowLogo) {
          Console.WriteLine(CommandLineOptions.Clo.Version);
        }
        if (CommandLineOptions.Clo.ShowEnv == CommandLineOptions.ShowEnvironment.Always) {
          Console.WriteLine("---Command arguments");
          foreach (string arg in args) {
            Contract.Assert(arg != null);
            Console.WriteLine(arg);
          }

          Console.WriteLine("--------------------");
        }

        Helpers.ExtraTraceInformation("Becoming sentient");

        List<string> fileList = new List<string>();
        foreach (string file in CommandLineOptions.Clo.Files) {
          string extension = Path.GetExtension(file);
          if (extension != null) {
            extension = extension.ToLower();
          }
          if (extension == ".txt") {
            StreamReader stream = new StreamReader(file);
            string s = stream.ReadToEnd();
            fileList.AddRange(s.Split(new char[3] { ' ', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries));
          }
          else {
            fileList.Add(file);
          }
        }
        foreach (string file in fileList) {
          Contract.Assert(file != null);
          string extension = Path.GetExtension(file);
          if (extension != null) {
            extension = extension.ToLower();
          }
          if (extension != ".bpl") {
            ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
            exitCode = 1;
            goto END;
          }
        }
        exitCode = ProcessFiles(fileList);

      END:
        if (CommandLineOptions.Clo.XmlSink != null) {
          CommandLineOptions.Clo.XmlSink.Close();
        }
        if (CommandLineOptions.Clo.Wait) {
          Console.WriteLine("Press Enter to exit.");
          Console.ReadLine();
        }

        Environment.Exit(exitCode);

      }
      catch (Exception e) {
        Console.Error.WriteLine("GPUVerify: an internal error has occurred.  Please report this problem to the development team");
        Environment.Exit(1);
      }
    }

    public static void ErrorWriteLine(string s) {
      Contract.Requires(s != null);
      ConsoleColor col = Console.ForegroundColor;
      Console.ForegroundColor = ConsoleColor.DarkGray;
      Console.Error.WriteLine(s);
      Console.ForegroundColor = col;
    }

    private static void ErrorWriteLine(string locInfo, string message, ErrorMsgType msgtype)
    {
      Contract.Requires(message != null);
      ConsoleColor col = Console.ForegroundColor;
      if (!String.IsNullOrEmpty(locInfo))
      {
        Console.Error.Write(locInfo + " ");
      }

      switch (msgtype)
      {
        case ErrorMsgType.Error:
          Console.ForegroundColor = ConsoleColor.Red;
          Console.Error.Write("error: ");
          break;
        case ErrorMsgType.Note:
          Console.ForegroundColor = ConsoleColor.DarkYellow;
          Console.Error.Write("note: ");
          break;
        case ErrorMsgType.NoError:
        default:
          break;
      }
        

      Console.ForegroundColor = col;
      Console.Error.WriteLine(message);
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

    enum FileType {
      Unknown,
      Cil,
      Bpl,
      Dafny
    };

    enum ErrorMsgType
    {
      Error,
      Note, 
      NoError
    };

    enum RaceType
    {
      WW,
      RW,
      WR
    };

    // Returns 0 if there were no errors, otherwise non-zero
    static int ProcessFiles(List<string> fileNames)
    {
      Contract.Requires(cce.NonNullElements(fileNames));
      using (XmlFileScope xf = new XmlFileScope(CommandLineOptions.Clo.XmlSink, fileNames[fileNames.Count - 1])) {
        //BoogiePL.Errors.count = 0;
        Program program = ParseBoogieProgram(fileNames, false);
        if (program == null)
          return 1;
        if (CommandLineOptions.Clo.PrintFile != null) {
          PrintBplFile(CommandLineOptions.Clo.PrintFile, program, false);
        }

        PipelineOutcome oc = ResolveAndTypecheck(program, fileNames[fileNames.Count - 1]);
        if (oc != PipelineOutcome.ResolvedAndTypeChecked)
          return 1;
        //BoogiePL.Errors.count = 0;

        // Do bitvector analysis
        if (CommandLineOptions.Clo.DoBitVectorAnalysis) {
          Microsoft.Boogie.BitVectorAnalysis.DoBitVectorAnalysis(program);
          PrintBplFile(CommandLineOptions.Clo.BitVectorAnalysisOutputBplFile, program, false);
          return 1;
        }

        if (CommandLineOptions.Clo.PrintCFGPrefix != null) {
          foreach (var impl in program.TopLevelDeclarations.OfType<Implementation>()) {
            using (StreamWriter sw = new StreamWriter(CommandLineOptions.Clo.PrintCFGPrefix + "." + impl.Name + ".dot")) {
              sw.Write(program.ProcessLoops(impl).ToDot());
            }
          }
        }

        EliminateDeadVariablesAndInline(program);

        int errorCount, verified, inconclusives, timeOuts, outOfMemories;
        oc = InferAndVerify(program, out errorCount, out verified, out inconclusives, out timeOuts, out outOfMemories);
        switch (oc) {
          case PipelineOutcome.Done:
          case PipelineOutcome.VerificationCompleted:
            WriteTrailer(verified, errorCount, inconclusives, timeOuts, outOfMemories);
            break;
          default:
            break;
        }

        return errorCount + inconclusives + timeOuts + outOfMemories;

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


    static bool ProgramHasDebugInfo(Program program) {
      Contract.Requires(program != null);
      // We inspect the last declaration because the first comes from the prelude and therefore always has source context.
      return program.TopLevelDeclarations.Count > 0 &&
          ((cce.NonNull(program.TopLevelDeclarations)[program.TopLevelDeclarations.Count - 1]).tok.IsValid);
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
      }
      else {
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



    static void ReportBplError(Absy node, string message, bool error, bool showBplLocation)
    {
      SourceLocationInfo sli = new SourceLocationInfo(GetAttributes(node), node.tok);

      string locinfo = null;

      if (showBplLocation)
      {
        locinfo = "File: \t"  + sli.GetFile() +
                  "\nLine:\t" + sli.GetLine() +
                  "\nCol:\t"  + sli.GetColumn() + "\n";
      }
      Contract.Requires(message != null);
      Contract.Requires(node != null);
      IToken tok = node.tok;
      if (error)
      {
        ErrorWriteLine(message);
      }
      else
      {
        Console.WriteLine(message);
      }
      if (!string.IsNullOrEmpty(locinfo))
      {
        ErrorWriteLine(locinfo);
      }
      else
      {
        ErrorWriteLine("Sourceloc info not found for: {0}({1},{2})\n", tok.filename, tok.line, tok.col);
      }
    }

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
        } catch (IOException e) {
          ErrorWriteLine("Error opening file \"{0}\": {1}", bplFileName, e.Message);
          okay = false;
          continue;
        }
        if (program == null) {
          program = programSnippet;
        } else if (programSnippet != null) {
          program.TopLevelDeclarations.AddRange(programSnippet.TopLevelDeclarations);
        }
      }
      if (!okay) {
        return null;
      } else if (program == null) {
        return new Program();
      } else {
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
              if (error is CallCounterexample)
              {
                CallCounterexample err = (CallCounterexample)error;
                if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "barrier_divergence"))
                {
                  ReportBarrierDivergence(err.FailingCall);
                }
                else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "race"))
                {
                  int byteOffset = -1, elemOffset = -1, elemWidth = -1;
                  string thread1 = null, thread2 = null, group1 = null, group2 = null, arrName = null;
                  //TODO: make this a command line argument.
                  bool detailedTrace = false;
                  string threadOneFailAccess = null, threadTwoFailAccess = null;

                  Variable offsetVar = ExtractOffsetVar(err.FailingRequires.Condition as NAryExpr);
                  /* Right now the offset incarnation that is extracted is the penultimate one if
                   * there is more than one incarnation, or the last one if there is only one incarnation.
                   * TODO: In future, we should know the exact incarnation to extract. This information is 
                   * available when the CallCounterexample is created in VC.cs AssertCmdToCounterexample() (line 2405)
                   * The condition of the AssertRequiresCmd contains the incarnation information, so this should be passed 
                   * on to the Requires of the created CallCounterexample. This can either be done by replacing the condition
                   * of the Requires with the condition from AssertRequiresCmd (containing incarnation information) or creating 
                   * a separate field in the Requires to store this original condition.
                   */
                  Model.Func offsetFunc = ExtractIncarnationFromModel(error.Model, offsetVar.Name);
                  GetInfoFromVarAndFunc(offsetVar.Attributes, offsetFunc, out byteOffset, out elemOffset, out elemWidth, out arrName);

                  GetThreadsAndGroupsFromModel(err.Model, out thread1, out thread2, out group1, out group2, true);

                  if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_read"))
                  {
                    err.FailingRequires.Attributes = GetSourceLocInfo(error, "WRITE");
                    threadOneFailAccess = "WRITE";
                    threadTwoFailAccess = "READ";
                    ReportRace(err.FailingCall, err.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WR);
                  }
                  else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "read_write"))
                  {
                    err.FailingRequires.Attributes = GetSourceLocInfo(error, "READ");
                    threadOneFailAccess = "READ";
                    threadTwoFailAccess = "WRITE";
                    ReportRace(err.FailingCall, err.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.RW);

                  }
                  else if (QKeyValue.FindBoolAttribute(err.FailingRequires.Attributes, "write_write"))
                  {
                    err.FailingRequires.Attributes = GetSourceLocInfo(error, "WRITE");
                    threadOneFailAccess = "WRITE";
                    threadTwoFailAccess = "WRITE";
                    ReportRace(err.FailingCall, err.FailingRequires, thread1, thread2, group1, group2, arrName, byteOffset, RaceType.WW);
                  }

                  if (detailedTrace)
                  {
                    string fname = QKeyValue.FindStringAttribute(err.FailingCall.Attributes, "fname");
                    Debug.Assert(!String.IsNullOrEmpty(fname));
                    int colOneSize = fname.Length + 11;
                    string colTwoHeader = " T " + GetThreadOne(error.Model, false) + " G " + GetGroupOne(error.Model, false) + " ";
                    string colThreeHeader = " T " + GetThreadTwo(error.Model, false) + " G " + GetGroupTwo(error.Model, false) + " ";
                    
                    int colTwoSize   = colTwoHeader.Length + 2;
                    int colThreeSize = colTwoHeader.Length + 2;
                    
                    PrintDetailedTraceHeader(colTwoHeader, colThreeHeader, colOneSize, colTwoSize, colThreeSize);
                    PrintDetailedTrace(err, colOneSize, colTwoSize, colThreeSize, elemOffset, elemWidth, threadOneFailAccess, threadTwoFailAccess);
                  }
                }
                else
                {
                  ReportRequiresFailure(err.FailingCall, err.FailingRequires);
                }
              }
              else if (error is ReturnCounterexample)
              {
                ReturnCounterexample err = (ReturnCounterexample)error;
                ReportEnsuresFailure(err.FailingEnsures);
              }
              else
              {
                AssertCounterexample err = (AssertCounterexample)error;
                if (err.FailingAssert is LoopInitAssertCmd)
                {
                  ReportInvariantEntryFailure(err.FailingAssert);
                }
                else if (err.FailingAssert is LoopInvMaintainedAssertCmd)
                {
                  ReportInvariantMaintedFailure(err.FailingAssert);
                }
                else
                {
                  ReportFailingAssert(err.FailingAssert);
                }
              }
              errorCount++;
            }
            //}
            Inform(String.Format("{0}error{1}", timeIndication, errors.Count == 1 ? "" : "s"));
          }
          break;
      }
    }

    private static void PrintDetailedTraceHeader(string colTwoHeader, string colThreeHeader, int locationColSize, int colTwoSize, int colThreeSize)
    {
      Console.Write("\nLocation");
      PrintNChars(locationColSize - "Location".Length, ' ');
      Console.WriteLine("|{0}|{1}", colTwoHeader + Chars(colTwoSize - colTwoHeader.Length, ' '), colThreeHeader + Chars(colThreeSize - colThreeHeader.Length, ' '));
      PrintNChars(locationColSize, '-');
      Console.Write("|");
      PrintNChars(colTwoSize, '-');
      Console.Write("|");
      PrintNChars(colThreeSize, '-');
      Console.WriteLine("");
    }

    private static void PrintNChars(int n, char c)
    {
      for (int i = 0; i < n; ++i)
      {
        Console.Write(c);
      }
    }

    private static string Chars(int n, char c)
    {
      return new String(c, n);
    }


    /// <summary>
    /// Given a resolved and type checked Boogie program, infers invariants for the program
    /// and then attempts to verify it.  Returns:
    ///  - Done if command line specified no verification
    ///  - FatalError if a fatal error occurred, in which case an error has been printed to console
    ///  - VerificationCompleted if inference and verification completed, in which the out
    ///    parameters contain meaningful values
    /// </summary>
    static PipelineOutcome InferAndVerify(Program program,
                                           out int errorCount, out int verified, out int inconclusives, out int timeOuts, out int outOfMemories) {
      Contract.Requires(program != null);
      Contract.Ensures(0 <= Contract.ValueAtReturn(out inconclusives) && 0 <= Contract.ValueAtReturn(out timeOuts));

      errorCount = verified = inconclusives = timeOuts = outOfMemories = 0;

      // ---------- Infer invariants --------------------------------------------------------

      // Abstract interpretation -> Always use (at least) intervals, if not specified otherwise (e.g. with the "/noinfer" switch)
      if (CommandLineOptions.Clo.UseAbstractInterpretation) {
        if (!CommandLineOptions.Clo.Ai.J_Intervals && !CommandLineOptions.Clo.Ai.J_Trivial) {
          // use /infer:j as the default
          CommandLineOptions.Clo.Ai.J_Intervals = true;
        }
      }
      Microsoft.Boogie.AbstractInterpretation.NativeAbstractInterpretation.RunAbstractInterpretation(program);

      if (CommandLineOptions.Clo.LoopUnrollCount != -1) {
        program.UnrollLoops(CommandLineOptions.Clo.LoopUnrollCount);
      }

      Dictionary<string, Dictionary<string, Block>> extractLoopMappingInfo = null;
      if (CommandLineOptions.Clo.ExtractLoops)
      {
        extractLoopMappingInfo = program.ExtractLoops();
      }

      if (CommandLineOptions.Clo.PrintInstrumented) {
        program.Emit(new TokenTextWriter(Console.Out));
      }

      if (CommandLineOptions.Clo.ExpandLambdas) {
        LambdaHelper.ExpandLambdas(program);
        //PrintBplFile ("-", program, true);
      }

      // ---------- Verify ------------------------------------------------------------

      if (!CommandLineOptions.Clo.Verify) {
        return PipelineOutcome.Done;
      }

      #region Run Houdini and verify
      if (CommandLineOptions.Clo.ContractInfer) {
        Houdini.Houdini houdini = new Houdini.Houdini(program);
        Houdini.HoudiniOutcome outcome = houdini.PerformHoudiniInference();
        if (CommandLineOptions.Clo.PrintAssignment) {
          Console.WriteLine("Assignment computed by Houdini:");
          foreach (var x in outcome.assignment) {
            Console.WriteLine(x.Key + " = " + x.Value);
          }
        }
        if (CommandLineOptions.Clo.Trace)
        {
          int numTrueAssigns = 0;
          foreach (var x in outcome.assignment) {
            if (x.Value)
              numTrueAssigns++;
          }
          Console.WriteLine("Number of true assignments = " + numTrueAssigns);
          Console.WriteLine("Number of false assignments = " + (outcome.assignment.Count - numTrueAssigns));
          Console.WriteLine("Prover time = " + Houdini.HoudiniSession.proverTime);
          Console.WriteLine("Unsat core prover time = " + Houdini.HoudiniSession.unsatCoreProverTime);
          Console.WriteLine("Number of prover queries = " + Houdini.HoudiniSession.numProverQueries);
          Console.WriteLine("Number of unsat core prover queries = " + Houdini.HoudiniSession.numUnsatCoreProverQueries);
          Console.WriteLine("Number of unsat core prunings = " + Houdini.HoudiniSession.numUnsatCorePrunings);
        }

        foreach (Houdini.VCGenOutcome x in outcome.implementationOutcomes.Values) {
          ProcessOutcome(x.outcome, x.errors, "", ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);
        }
        //errorCount = outcome.ErrorCount;
        //verified = outcome.Verified;
        //inconclusives = outcome.Inconclusives;
        //timeOuts = outcome.TimeOuts;
        //outOfMemories = 0;
        return PipelineOutcome.Done;
      }
      #endregion

      #region Verify each implementation

      ConditionGeneration vcgen = null;
      try {
        if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
          Debug.Assert(false);
        } else if(CommandLineOptions.Clo.StratifiedInlining > 0) {
          vcgen = new StratifiedVCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend);
        } else {
          vcgen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend);
        }
      } catch (ProverException e) {
        ErrorWriteLine("Fatal Error: ProverException: {0}", e);
        return PipelineOutcome.FatalError;
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
          if (CommandLineOptions.Clo.Trace || CommandLineOptions.Clo.TraceProofObligations || CommandLineOptions.Clo.XmlSink != null) {
            start = DateTime.UtcNow;
            if (CommandLineOptions.Clo.Trace || CommandLineOptions.Clo.TraceProofObligations) {
              Console.WriteLine();
              Console.WriteLine("Verifying {0} ...", impl.Name);
            }
            if (CommandLineOptions.Clo.XmlSink != null) {
              CommandLineOptions.Clo.XmlSink.WriteStartMethod(impl.Name, start);
            }
          }

          VCGen.Outcome outcome;
          try {
            if (CommandLineOptions.Clo.inferLeastForUnsat != null) {
              var svcgen = vcgen as VC.StratifiedVCGen;
              Contract.Assert(svcgen != null);
              var ss = new HashSet<string>();
              foreach (var tdecl in program.TopLevelDeclarations) {
                var c = tdecl as Constant;
                if (c == null || !c.Name.StartsWith(CommandLineOptions.Clo.inferLeastForUnsat)) continue;
                ss.Add(c.Name);
              }
              outcome = svcgen.FindLeastToVerify(impl, ref ss);
              errors = new List<Counterexample>();
              Console.Write("Result: ");
              foreach (var s in ss) {
                Console.Write("{0} ", s);
              }
              Console.WriteLine();
            }
            else {
              outcome = vcgen.VerifyImplementation(impl, out errors);
              if (CommandLineOptions.Clo.ExtractLoops && vcgen is VCGen && errors != null) {
                for (int i = 0; i < errors.Count; i++) {
                  errors[i] = (vcgen as VCGen).extractLoopTrace(errors[i], impl.Name, program, extractLoopMappingInfo);
                }
              }
            }
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
          } else if (CommandLineOptions.Clo.TraceProofObligations) {
              int poCount = vcgen.CumulativeAssertionCount - prevAssertionCount;
            timeIndication = string.Format("  [{0} proof obligation{1}]  ", poCount, poCount == 1 ? "" : "s");
          }

          ProcessOutcome(outcome, errors, timeIndication, ref errorCount, ref verified, ref inconclusives, ref timeOuts, ref outOfMemories);

          if (CommandLineOptions.Clo.XmlSink != null) {
            CommandLineOptions.Clo.XmlSink.WriteEndMethod(outcome.ToString().ToLowerInvariant(), end, elapsed);
          }
          if (outcome == VCGen.Outcome.Errors || CommandLineOptions.Clo.Trace) {
            Console.Out.Flush();
          }
        }
      }

      vcgen.Close();
      cce.NonNull(CommandLineOptions.Clo.TheProverFactory).Close();


      #endregion

      return PipelineOutcome.VerificationCompleted;
    }

    private static void AddPadding(ref string string1, ref string string2)
    {
      if (string1.Length < string2.Length) 
      {
        for (int i = (string2.Length - string1.Length); i > 0; --i)
        {
          string1 += " ";
        }
      }
      else
      {
        for (int i = (string1.Length - string2.Length); i > 0; --i)
        {
          string2 += " ";
        }
      }
    }

    private static string TrimLeadingSpaces(string s1, int noOfSpaces)
    {
      if (String.IsNullOrWhiteSpace(s1))
      {
        return s1;
      }

      int index;
      for (index = 0; (index + 1) < s1.Length && Char.IsWhiteSpace(s1[index]); ++index);
      string returnString = s1.Substring(index);
      for (int i = noOfSpaces; i > 0; --i)
      {
        returnString = " " + returnString;
      }
      return returnString;
    }

    static QKeyValue GetAttributes(Absy a)
    {
      if (a is PredicateCmd)
      {
        return (a as PredicateCmd).Attributes;
      }
      else if (a is Requires)
      {
        return (a as Requires).Attributes;
      }
      else if (a is Ensures)
      {
        return (a as Ensures).Attributes;
      }
      else if (a is CallCmd)
      {
        return (a as CallCmd).Attributes;
      }
      //Debug.Assert(false);
      return null;
    }

    static Model.Func ExtractIncarnationFromModel(Model m, string varName)
    {
      try {
        Model.Func lastFunc = null;
        Model.Func penulFunc = null;
        int currIncarnationNo = -1;
        foreach (Model.Func f in m.Functions) {
          if (f.Name.Contains(varName)) {
            string[] tokens = Regex.Split(f.Name, "@");
            if (tokens.Length == 2 && System.Convert.ToInt32(tokens[1]) > currIncarnationNo) {
              penulFunc = lastFunc;
              lastFunc = f;
              currIncarnationNo = System.Convert.ToInt32(tokens[1]);
            }
          }
        }
        return (penulFunc == null) ? lastFunc : penulFunc;
      }
      catch (Exception) {
        return null;
      }
    }

    static void PrintDetailedTrace(CallCounterexample err, int colOneLength, int colTwoLength, int colThreeLength, int failElemOffset, int elemWidth, string threadOneFailAccess, string threadTwoFailAccess)
    {
      Model model = err.Model;
      BlockSeq trace = err.Trace;

      int failCallLineNo = QKeyValue.FindIntAttribute(err.FailingCall.Attributes, "line", -1);
      int failCallColNo  = QKeyValue.FindIntAttribute(err.FailingCall.Attributes, "col", -1);
      int failReqLineNo  = QKeyValue.FindIntAttribute(err.FailingRequires.Attributes, "line", -1);
      int failReqColNo   = QKeyValue.FindIntAttribute(err.FailingRequires.Attributes, "col", -1);

      int lineno = -1;
      int colno = -1;

      bool checkCallExpected = false;
      bool colTwoFail = false;
      bool colThreeFail = false;

      string colOne   = null;
      string colTwo   = null;
      string colThree = null;

      foreach (Block b in trace)
      {
        if (Regex.IsMatch(b.ToString(), @"inline[$]_LOG_(READ|WRITE)_[$]+\w+[$][0-9]+[$]_LOG_(READ|WRITE)"))
        {
          int elemOffset = -1;
          string arrName;

          foreach (Cmd c in b.Cmds)
          {
            string access = (Regex.IsMatch(c.ToString(), @"inline[$]_LOG_READ")) ? "READ" : "WRITE";

            if (Regex.IsMatch(c.ToString(), "assume _" + access + "_OFFSET")) 
            {
              elemOffset = ExtractOffsetFromExpr(model, (((c as PredicateCmd).Expr as NAryExpr).Args[1] as NAryExpr).Args[1]);
            }
            else if (Regex.IsMatch(c.ToString(), "assume _" + access + "_SOURCE"))
            {
              arrName = ExtractArrName(c.ToString());
              string enParamName = ExtractEnabledArg((c as AssumeCmd).Expr).Name;

              Model.Func enFunc = model.TryGetFunc(enParamName);
              Debug.Assert(enFunc != null, "PrintDetailedTrace(): could not get enParamName from model: " + enParamName);
              Debug.Assert(enFunc.AppCount == 1, "PrintDetailedTrace(): enabled parameter function has more that one application.");
              bool enabled = (enFunc.Apps.ElementAt(0).Result as Model.Boolean).Value;
              int sourceLocLineNo = (ExtractSourceLineArg((c as AssumeCmd).Expr).Val as BvConst).Value.ToIntSafe;

              string sourceLocLine = SourceLocationInfo.FetchCodeLine(GetSourceLocFileName(), sourceLocLineNo + 1);

              string[] slocTokens = Regex.Split(sourceLocLine, "#");
              lineno = System.Convert.ToInt32(slocTokens[0]);
              colno = System.Convert.ToInt32(slocTokens[1]);
              string fname = slocTokens[2];

              colOne = fname + ":" + lineno + ":" + colno + ":";
              colTwo = enabled ? " " + access.ToLower() + "s ((char*)" + arrName + ")" + "[" + ((elemOffset == -1) ? "?" : "" + CalculateByteOffset(elemOffset, elemWidth)) + "]" : "";
              colTwoFail = elemOffset == failElemOffset && lineno == failReqLineNo && colno == failReqColNo && access == threadOneFailAccess;
            }
          }
        }
        else if (Regex.IsMatch(b.ToString(), @"inline[$]_LOG_(READ|WRITE)_[$]+\w+[$][0-9]+[$]Return"))
        {
          // Assuming that a LOG call will always be immediately followed by a CHECK call.
          checkCallExpected = true;
          continue;
        }
        else if (checkCallExpected)
        {
          foreach (Cmd c in b.Cmds)
          {
            if (Regex.IsMatch(c.ToString(), @"assert[\s]+[!]\(\w+[$][0-9]+(@[0-9]+|[\s]+)[\s]*&&[\s]+_(WRITE|READ)_HAS_OCCURRED"))
            {
              string access = Regex.IsMatch((c as AssertRequiresCmd).Call.callee, "_CHECK_READ_") ? "READ" : "WRITE";

              string[] tokens = Regex.Split(c.ToString(), "&&");
              string arrName = ExtractArrName(tokens[1]);
              string enParamName = ExtractEnabledParameterName((c as AssertRequiresCmd).Expr);

              Model.Func enFunc = model.TryGetFunc(enParamName);
              Debug.Assert(enFunc != null, "PrintDetailedTrace(): could not get enParamName from model: " + enParamName);
              Debug.Assert(enFunc.AppCount == 1, "PrintDetailedTrace(): enabled parameter function has more that one application.");
              bool enabled = (enFunc.Apps.ElementAt(0).Result as Model.Boolean).Value;

              Expr offsetArg = ExtractOffsetArg((c as AssertRequiresCmd).Expr, "_" + (c.ToString().Contains("WRITE") ? "WRITE" : "READ") + "_OFFSET_");



              int elemOffset = ExtractOffsetFromExpr(model, offsetArg);
              colThree = enabled ? " " + access.ToLower() + "s ((char*)" + arrName + ")" + "[" + ((elemOffset == -1) ? "?" : "" + CalculateByteOffset(elemOffset, elemWidth)) + "]" : "";
              colThreeFail = elemOffset == failElemOffset && lineno == failCallLineNo && colno == failCallColNo && access == threadTwoFailAccess;

              checkCallExpected = false;
              PrintDetailedTraceLine(colOne, colTwo, colThree, colOneLength, colTwoLength, colThreeLength, colTwoFail, colThreeFail);
              break;
            }
          }
        }
      }
    }

    private static LiteralExpr ExtractSourceLineArg(Expr e)
    {
      var visitor = new GetThenOfIfThenElseVisitor();
      visitor.VisitExpr(e);
      return visitor.getResult();
    }

    private static Expr ExtractOffsetArg(Expr e, string pattern)
    {
      var visitor = new GetRHSOfEqualityVisitor(pattern);
      visitor.VisitExpr(e);
      return visitor.getResult();
    }

    private static IdentifierExpr ExtractEnabledArg(Expr e)
    {
      var visitor = new GetIfOfIfThenElseVisitor();
      visitor.VisitExpr(e);
      return visitor.getResult();
    }

    private static string ExtractEnabledParameterName(Expr e)
    {
      if (e is IdentifierExpr)
      {
        return ((IdentifierExpr)e).Name;
      }
      else
      {
        Debug.Assert(e is NAryExpr);
        return ExtractEnabledParameterName(((NAryExpr)e).Args[0]);
      }
    }

    private static int ExtractOffsetFromExpr(Model model, Expr offsetArg)
    {
      if (offsetArg is LiteralExpr)
      {
        return ((offsetArg as LiteralExpr).Val as BvConst).Value.ToIntSafe;
      }
      else if (offsetArg is IdentifierExpr)
      {
        string offsetname = (offsetArg as IdentifierExpr).Name;
        Model.Func fo = model.TryGetFunc(offsetname);
        Debug.Assert(fo != null, "ExtractOffsetFromExpr(): could not get value for the following from model: " + offsetname);
        Debug.Assert(fo.AppCount == 1, "ExtractOffsetFromExpr(): the following function has more than one application: " + offsetname);
        return (fo.Apps.ElementAt(0).Result as Model.BitVector).AsInt();

      }
      else
      {
        return -1;
      }
    }

    private static void PrintDetailedTraceLine(string colOne, string colTwo, string colThree, int colOneLength, int colTwoLength, int colThreeLength, bool colTwoError, bool colThreeError)
    {
      Contract.Requires(colOneLength >= colOne.Length && colTwoLength >= colTwo.Length && colThreeLength >= colThree.Length);
      Console.Write("{0}|", colOne + Chars(colOneLength - colOne.Length, ' '));

      ConsoleColor col = Console.ForegroundColor;
      if (colTwoError) 
      {
        Console.ForegroundColor = ConsoleColor.Red;
      }
      Console.Write("{0}", colTwo + Chars(colTwoLength - colTwo.Length, ' '));
      Console.ForegroundColor = col;
      Console.Write('|');

      if (colThreeError)
      {
        Console.ForegroundColor = ConsoleColor.Red;
      }
      Console.WriteLine("{0}", colThree + Chars(colThreeLength - colThree.Length, ' '));
      Console.ForegroundColor = col;

    }

   private static string GetSourceLocFileName()
    {
      return GetFilenamePathPrefix() + GetFileName() + ".loc";
    }

    private static string GetFileName()
    {
      return Path.GetFileNameWithoutExtension(CommandLineOptions.Clo.Files[0]);
    }

    private static string GetFilenamePathPrefix()
    {
      string directoryName = Path.GetDirectoryName(CommandLineOptions.Clo.Files[0]);
      return ((!String.IsNullOrEmpty(directoryName) && directoryName != ".") ? (directoryName + Path.DirectorySeparatorChar) : "");
    }

    private static string GetCorrespondingThreadTwoName(string threadOneName)
    {
      return threadOneName.Replace("$1", "$2");
    }
    static QKeyValue CreateSourceLocQKV(int line, int col, string fname, string dir)
    {
      QKeyValue dirkv = new QKeyValue(Token.NoToken, "dir", new List<object>(new object[] { dir }), null);
      QKeyValue fnamekv = new QKeyValue(Token.NoToken, "fname", new List<object>(new object[] { fname }), dirkv);
      QKeyValue colkv = new QKeyValue(Token.NoToken, "col", new List<object>(new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(col)) }), fnamekv);
      QKeyValue linekv = new QKeyValue(Token.NoToken, "line", new List<object>(new object[] { new LiteralExpr(Token.NoToken, BigNum.FromInt(line)) }), colkv);
      return linekv;
    }

    static QKeyValue GetSourceLocInfo(Counterexample error, string AccessType) {
      try {
        string sourceVarName = null;
        int sourceLocLineNo = -1;

        foreach (Block b in error.Trace) {
          foreach (Cmd c in b.Cmds) {
            if (b.tok.val.Equals("_LOG_" + AccessType) && c.ToString().Contains(AccessType + "_SOURCE_")) {
              sourceVarName = Regex.Split(c.ToString(), " ")[1];
            }
          }
        }
        if (sourceVarName != null) {
          Model.Func f = error.Model.TryGetFunc(sourceVarName);
          if (f != null) {
            sourceLocLineNo = f.GetConstant().AsInt();
          }
        }

        if (sourceLocLineNo > 0) {
          // TODO: Make lines in .loc file be indexed from 1 for consistency.
          string fileLine = SourceLocationInfo.FetchCodeLine(GetSourceLocFileName(), sourceLocLineNo + 1);
          if (fileLine != null) {
            string[] slocTokens = Regex.Split(fileLine, "#");
            return CreateSourceLocQKV(
                    System.Convert.ToInt32(slocTokens[0]),
                    System.Convert.ToInt32(slocTokens[1]),
                    slocTokens[2],
                    slocTokens[3]);
          }
        }
        else {
          return null;
        }
        return null;
      }
      catch (Exception) {
        return null;
      }
    }

    static bool IsRepeatedKV(QKeyValue attrs, List<QKeyValue> alreadySeen)
    {
      return false;
      /*
      if (attrs == null)
      {
        return false;
      }
      string key = null;
      foreach (QKeyValue qkv in alreadySeen)
      {
        QKeyValue kv = qkv.Clone() as QKeyValue;
        if (kv.Params.Count != attrs.Params.Count) 
        {
          return false;
        }
        for (; kv != null; kv = kv.Next) {
          key = kv.Key;
          if (key != "thread") {
            if (kv.Params.Count == 0)
            {
              if (QKeyValue.FindBoolAttribute(attrs, key))
              {
                continue;
              }
              else
              {
                return false;
              }

            }
            else if (kv.Params[0] is LiteralExpr)
            { // int
              LiteralExpr l = kv.Params[0] as LiteralExpr;
              int i = l.asBigNum.ToIntSafe;
              if (QKeyValue.FindIntAttribute(attrs, key, -1) == i)
              {
                continue;
              }
              else
              {
                return false;
              }
            }
            else if (kv.Params[0] is string)
            { // string
              string s = kv.Params[0] as string;
              if (QKeyValue.FindStringAttribute(attrs, key) == s)
              {
                continue;
              }
              else
              {
                return false;
              }
            }
            else
            {
              Debug.Assert(false);
              return false;
            }
          }
        }
        return true;
      }
      return false;
      */
    }

    private static void ReportFailingAssert(Absy node)
    {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);

      ErrorWriteLine(sli.ToString(), "this assertion might not hold", ErrorMsgType.Error);
      ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportInvariantMaintedFailure(Absy node)
    {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "loop invariant might not be maintained by the loop", ErrorMsgType.Error);
      ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportInvariantEntryFailure(Absy node)
    {
      Console.WriteLine("");
      SourceLocationInfo sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "loop invariant might not hold on entry", ErrorMsgType.Error);
      ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportEnsuresFailure(Absy node)
    {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "postcondition might not hold on all return paths", ErrorMsgType.Error);
      ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRace(Absy callNode, Absy reqNode, string thread1, string thread2, string group1, string group2, string arrName, int byteOffset, RaceType raceType)
    {
      Console.WriteLine("");
      string locinfo1 = null, locinfo2 = null, raceName, access1, access2;

      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      switch (raceType)
      {
        case RaceType.RW:
          raceName = "read-write";
          access1 = "read";
          access2 = "write";
          break;
        case RaceType.WR:
          raceName = "write-read";
          access1 = "write";
          access2 = "read";
          break;
        case RaceType.WW:
          raceName = "write-write";
          access1 = "write";
          access2 = "write";
          break;
        default:
          raceName = null;
          access1 = null;
          access2 = null;
          Debug.Assert(false, "ReportRace(): Reached default case in switch over raceType.");
          break;
      }
      ErrorWriteLine(CallSLI.GetFile() + ":", "possible " + raceName + " race on ((char*)" + arrName + ")[" + byteOffset + "]:\n", ErrorMsgType.Error);

      locinfo1 = CallSLI.ToString();
      locinfo2 = RequiresSLI.ToString();

      AddPadding(ref locinfo1, ref locinfo2);

      ErrorWriteLine(locinfo1, access2 + " by thread " + thread2 + " group " + group2, ErrorMsgType.NoError);
      ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine() + "\n", 2));


      ErrorWriteLine(locinfo2, access1 + " by thread " + thread1 + " group " + group1, ErrorMsgType.NoError);
      ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine() + "\n", 2));
    }

    private static void ReportBarrierDivergence(Absy node)
    {
      Console.WriteLine("");
      var sli = new SourceLocationInfo(GetAttributes(node), node.tok);
      ErrorWriteLine(sli.ToString(), "barrier may be reached by non-uniform control flow", ErrorMsgType.Error);
      ErrorWriteLine(sli.FetchCodeLine());
    }

    private static void ReportRequiresFailure(Absy callNode, Absy reqNode)
    {
      Console.WriteLine("");
      var CallSLI = new SourceLocationInfo(GetAttributes(callNode), callNode.tok);
      var RequiresSLI = new SourceLocationInfo(GetAttributes(reqNode), reqNode.tok);

      ErrorWriteLine(CallSLI.ToString(), "a precondition for this call might not hold", ErrorMsgType.Error);
      ErrorWriteLine(TrimLeadingSpaces(CallSLI.FetchCodeLine(), 2));

      ErrorWriteLine(RequiresSLI.ToString(), "this is the precondition that might not hold", ErrorMsgType.Note);
      ErrorWriteLine(TrimLeadingSpaces(RequiresSLI.FetchCodeLine(), 2));
    }

    private static void GetThreadsAndGroupsFromModel(Model model, out string thread1, out string thread2, out string group1, out string group2, bool withSpaces)
    {
      thread1 = GetThreadOne(model, withSpaces);
      thread2 = GetThreadTwo(model, withSpaces);
      group1 = GetGroupOne(model, withSpaces);
      group2 = GetGroupTwo(model, withSpaces);
    }

    private static string GetGroupTwo(Model model, bool withSpaces)
    {
      return "("
             + GetGidX2(model)
             + "," + (withSpaces ? " " : "")
             + GetGidY2(model)
             + "," + (withSpaces ? " " : "")
             + GetGidZ2(model)
             + ")";
    }

    private static int GetGidZ2(Model model)
    {
      return model.TryGetFunc("group_id_z$2").GetConstant().AsInt();
    }

    private static int GetGidY2(Model model)
    {
      return model.TryGetFunc("group_id_y$2").GetConstant().AsInt();
    }

    private static int GetGidX2(Model model)
    {
      return model.TryGetFunc("group_id_x$2").GetConstant().AsInt();
    }

    private static string GetGroupOne(Model model, bool withSpaces)
    {
      return "("
             + GetGidX1(model)
             + "," + (withSpaces ? " " : "")
             + GetGidY1(model)
             + "," + (withSpaces ? " " : "")
             + GetGidZ1(model) 
             + ")";
    }

    private static int GetGidZ1(Model model)
    {
      return model.TryGetFunc("group_id_z$1").GetConstant().AsInt();
    }

    private static int GetGidY1(Model model)
    {
      return model.TryGetFunc("group_id_y$1").GetConstant().AsInt();
    }

    private static int GetGidX1(Model model)
    {
      return model.TryGetFunc("group_id_x$1").GetConstant().AsInt();
    }

    private static string GetThreadTwo(Model model, bool withSpaces)
    {
      return "("
             + GetLidX2(model)
             + "," + (withSpaces ? " " : "")
             + GetLidY2(model)
             + "," + (withSpaces ? " " : "")
             + GetLidZ2(model) 
             + ")";
    }

    private static int GetLidZ2(Model model)
    {
      return model.TryGetFunc("local_id_z$2").GetConstant().AsInt();
    }

    private static int GetLidY2(Model model)
    {
      return model.TryGetFunc("local_id_y$2").GetConstant().AsInt();
    }

    private static int GetLidX2(Model model)
    {
      return model.TryGetFunc("local_id_x$2").GetConstant().AsInt();
    }

    private static string GetThreadOne(Model model, bool withSpaces)
    {
      return "(" 
             + model.TryGetFunc("local_id_x$1").GetConstant().AsInt() 
             + "," + (withSpaces ? " " : "")
             + model.TryGetFunc("local_id_y$1").GetConstant().AsInt() 
             + "," + (withSpaces ? " " : "")
             + model.TryGetFunc("local_id_z$1").GetConstant().AsInt() 
             + ")";
    }

    private static void GetInfoFromVarAndFunc(QKeyValue attrs, Model.Func f, out int byteOffset, out int elemOffset, out int elemWidth, out string arrName)
    {
      if (attrs == null) {
        elemWidth = -1;
      }
      else {
        elemWidth = QKeyValue.FindIntAttribute(attrs, "elem_width", -1);
      }
      if (f == null) {
        elemOffset = -1;
        arrName = "<unknown array>";
      }
      else {
        try {
          elemOffset = f.Apps.FirstOrDefault().Result.AsInt();
        }
        catch (System.OverflowException e) {
          elemOffset = -1;
        }
        arrName = ExtractArrName(f.Name);
      }
      if (attrs == null || f == null) {
        byteOffset = -1;
      }
      else {
        byteOffset = CalculateByteOffset(elemOffset, elemWidth);
      }
    }

    private static int CalculateByteOffset(int elemOffset, int elemWidth)
    {
      return (elemOffset * elemWidth) / 8;
    }

    private static string ExtractArrName(string varName)
    {
      return Regex.Split(varName, "[$]+")[1];
    }

    private static Variable ExtractOffsetVar(NAryExpr expr)
    {
      foreach (Expr e in expr.Args)
      {
        if (e is NAryExpr && e.ToString().Contains("_OFFSET_"))
        {
          return ExtractOffsetVar(e as NAryExpr);
        }
        else if (e is IdentifierExpr && (e as IdentifierExpr).Name.Contains("_OFFSET_"))
        {
          return (e as IdentifierExpr).Decl;
        }
        else continue;
      }
      Debug.Assert(false, "GPUVerifyBoogieDriver: ExtractOffsetExpr() could not find _OFFSET expr.");
      return null;
    }

  }
}
