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
using System.Diagnostics.Contracts;
using System.Linq;
using System.Reflection;
using System.IO;
using Microsoft.Boogie;

using ConcurrentHoudini = Microsoft.Boogie.Houdini.ConcurrentHoudini;

namespace GPUVerify
{
  /// <summary>
  /// Utility class for GPUVerify.
  /// </summary>
  public static class GVUtil
  {

    /// <summary>
    /// Returns a Microsoft.Boogie.Houdini.ConcurrentHoudini.RefutedAnnotation object by iterating the
    /// TopLevelDeclarations of a Program using the specified strings.
    /// </summary>
    public static ConcurrentHoudini.RefutedAnnotation getRefutedAnnotation(Program program, string constant, string implementation)
    {
      ConcurrentHoudini.RefutedAnnotation ra = null;
      Variable variable = null;
      Implementation refutationSite = null;

      foreach (var v in program.TopLevelDeclarations.OfType<Variable>()) {
        if (v.Name.Equals(constant)) {
          variable = v;
          break;
        }
      }

      foreach (var r in program.TopLevelDeclarations.OfType<Implementation>()) {
        if (r.Name.Equals(implementation)) {
          refutationSite = r;
          break;
        }
      }

      ra = ConcurrentHoudini.RefutedAnnotation.BuildRefutedAssert(variable, refutationSite);

      return ra;
    }

    /// <summary>
    /// IO utility class for GPUVerify.
    /// </summary>
    public static class IO
    {
      public static void EmitProgram(Program prog, string filename, string extension = "bpl")
      {
        using (TokenTextWriter writer = new TokenTextWriter(filename + "." + extension, false)) {
          prog.Emit(writer);
        }
      }

      public static Program ParseBoogieProgram(IList<string> fileNames, bool suppressTraceOutput)
      {
        Contract.Requires(cce.NonNullElements(fileNames));

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
            errorCount = Parser.Parse(bplFileName, defines, out programSnippet);
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
            program.AddTopLevelDeclarations(programSnippet.TopLevelDeclarations);
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

      public static void PrintBplFile(string filename, Program program, bool allowPrintDesugaring)
      {
        Contract.Requires(program != null);
        Contract.Requires(filename != null);

        bool oldPrintDesugaring = CommandLineOptions.Clo.PrintDesugarings;
        if (!allowPrintDesugaring) {
          CommandLineOptions.Clo.PrintDesugarings = false;
        }
        using (TokenTextWriter writer = filename == "-" ?
               new TokenTextWriter("<console>", Console.Out, false) :
               new TokenTextWriter(filename, false)) {
          if (CommandLineOptions.Clo.ShowEnv != CommandLineOptions.ShowEnvironment.Never) {
            writer.WriteLine("// " + CommandLineOptions.Clo.Version);
            writer.WriteLine("// " + CommandLineOptions.Clo.Environment);
          }
          writer.WriteLine();
          program.Emit(writer);
        }
        CommandLineOptions.Clo.PrintDesugarings = oldPrintDesugaring;
      }

      public static void ReportBplError(Absy node, string message, bool error, bool showBplLocation)
      {
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

      public static void WriteTrailer(KernelAnalyser.ResultCounter result)
      {
        Contract.Requires(0 <= result.VerificationErrors && 0 <= result.Inconclusives && 0 <= result.TimeOuts && 0 <= result.OutOfMemories);

        //Console.WriteLine();
        if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed) {
          Console.Write("{0} finished with {1} credible, {2} doomed{3}",
            CommandLineOptions.Clo.DescriptiveToolName,
            result.Verified,
            result.VerificationErrors,
            result.VerificationErrors == 1 ? "" : "s");
        } else {
          Console.Write("{0} finished with {1} verified, {2} error{3}",
            CommandLineOptions.Clo.DescriptiveToolName,
            result.Verified,
            result.VerificationErrors,
            result.VerificationErrors == 1 ? "" : "s");
        }
        if (result.Inconclusives != 0) {
          Console.Write(", {0} inconclusive{1}", result.Inconclusives, result.Inconclusives == 1 ? "" : "s");
        }
        if (result.TimeOuts != 0) {
          Console.Write(", {0} time out{1}", result.TimeOuts, result.TimeOuts == 1 ? "" : "s");
        }
        if (result.OutOfMemories != 0) {
          Console.Write(", {0} out of memory", result.OutOfMemories);
        }
        Console.WriteLine();
        Console.Out.Flush();
      }

      public static void Inform(string s) {
        if (CommandLineOptions.Clo.Trace || CommandLineOptions.Clo.TraceProofObligations)
        {
          Console.WriteLine(s);
        }
      }

      public static void ErrorWriteLine(string s)
      {
        Contract.Requires(s != null);
        ConsoleColor col = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Error.WriteLine(s);
        Console.ForegroundColor = col;
      }

      public static void ErrorWriteLine(string format, params object[] args)
      {
        Contract.Requires(format != null);
        string s = string.Format(format, args);
        ErrorWriteLine(s);
      }

      public static void AdvisoryWriteLine(string format, params object[] args)
      {
        Contract.Requires(format != null);
        ConsoleColor col = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(format, args);
        Console.ForegroundColor = col;
      }

      public static void DumpExceptionInformation(Exception e)
      {

        if(e.ToString().Contains("An attempt was made to load an assembly from a network location")) {
          Console.Error.WriteLine();
          Console.Error.WriteLine("GPUVerify has had trouble loading one of its components due to security settings.");
          Console.Error.WriteLine();
          Console.Error.WriteLine("In order to run GPUVerify successfully you need to unblock the archive before unzipping it.");
          Console.Error.WriteLine();
          Console.Error.WriteLine("To do this:");
          Console.Error.WriteLine(" - Right click on the .zip file");
          Console.Error.WriteLine(" - Click \"Properties\"");
          Console.Error.WriteLine(" - At the bottom of the \"General\" tab you should see:");
          Console.Error.WriteLine("     Security: This file came from another computer and might be blocked");
          Console.Error.WriteLine("     to help protect this computer.");
          Console.Error.WriteLine(" - Click \"Unblock\"");
          Console.Error.WriteLine(" - Click \"OK\"");
          Console.Error.WriteLine("Once this is done, unzip GPUVerify afresh and this issue should be resolved.");
          Environment.Exit((int)ToolExitCodes.INTERNAL_ERROR);
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
        #endregion

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
        using (TokenTextWriter writer = new TokenTextWriter(DUMP_FILE, false)) {
          writer.Write("Exception ToString:");
          writer.Write("===================");
          writer.Write(e.ToString());
          writer.Close();
        }
        #endregion
      }
    }

    // TODO: Not working really yet ...
    // Performs deep copy of an object using reflection
    public static object DeepCopy(object obj)
    {
      if (obj == null) return null;

      System.Type type = obj.GetType();
      Console.WriteLine("***TYPE 0: {0}***", type.FullName);

      if (type.IsPrimitive || type.IsEnum || type == typeof(string)) {
        Console.WriteLine("***TYPE 1: {0}***", type.FullName);

        return obj;
      } else if (type.IsArray) {
        Console.WriteLine("***TYPE 2: {0}***", type.FullName);
        Console.WriteLine("***TYPE 2: {0}***", type.FullName.Replace("[]", string.Empty));

        System.Type elementType = System.Type.GetType(type.FullName.Replace("[]", string.Empty) + ", Core");
        Console.WriteLine("ELEMENTTYPE: " + elementType.FullName);
        var array = obj as Array;
        Array copied = Array.CreateInstance(elementType, array.Length);

        for (int i = 0; i < array.Length; i++) {
          copied.SetValue(DeepCopy(array.GetValue(i)), i);
        }

        return Convert.ChangeType(copied, obj.GetType());
      } else if (type.IsClass || type.IsValueType) {
        Console.WriteLine("***TYPE 3: {0}***", type.FullName);
        object copiedObject = Activator.CreateInstance(obj.GetType());
        FieldInfo[] fields = type.GetFields(BindingFlags.Public |
                                            BindingFlags.NonPublic |
                                            BindingFlags.Instance);

        foreach (FieldInfo field in fields) {
          object fieldValue = field.GetValue(obj);

          if (fieldValue != null)
            field.SetValue(copiedObject, DeepCopy(fieldValue));
        }

        return copiedObject;
      } else throw new ArgumentException("Unknown type");
    }

    public static string StripThreadIdentifier(string p, out int id)
    {
      if (p.EndsWith("$1") && !p.Equals("$1"))
      {
        id = 1;
        return p.Substring(0, p.Length - 2);
      }
      if (p.EndsWith("$2") && !p.Equals("$2"))
      {
        id = 2;
        return p.Substring(0, p.Length - 2);
      }

      id = 0;
      return p;
    }

    public static string StripThreadIdentifier(string p)
    {
      int id;
      return StripThreadIdentifier(p, out id);
    }

    public static Program GetFreshProgram(IList<string> fileNames, bool disableChecks, bool inline)
    {
      KernelAnalyser.PipelineOutcome oc;
      Program program = GVUtil.IO.ParseBoogieProgram(fileNames, false);
      if (program == null) Environment.Exit((int)ToolExitCodes.OTHER_ERROR);
      oc = KernelAnalyser.ResolveAndTypecheck(program, fileNames[fileNames.Count - 1]);
      if (oc != KernelAnalyser.PipelineOutcome.ResolvedAndTypeChecked)
        Environment.Exit((int)ToolExitCodes.OTHER_ERROR);

      if (disableChecks) {
        KernelAnalyser.DisableRaceChecking(program);
        KernelAnalyser.DisableBarrierDivergenceChecking(program);
        KernelAnalyser.DisableAssertions(program);
      }

      KernelAnalyser.EliminateDeadVariables(program);
      if (inline) KernelAnalyser.Inline(program);

      return program;
    }


  }
}
