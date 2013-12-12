//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Windows.Forms;
using Microsoft.Boogie;
using System.Diagnostics.Contracts;

namespace GPUVerify
{
    class GPUVerify
    {
        public static void Main(string[] args)
        {
          try {
            int showHelp = GPUVerifyVCGenCommandLineOptions.Parse(args);

            if (showHelp == -1) {
              GPUVerifyVCGenCommandLineOptions.Usage();
              System.Environment.Exit(0);
            }

            if (GPUVerifyVCGenCommandLineOptions.inputFiles.Count < 1) {
              Console.WriteLine("*** Error: No input files were specified.");
              Environment.Exit(1);
            }

            foreach (string file in GPUVerifyVCGenCommandLineOptions.inputFiles) {
              string extension = Path.GetExtension(file);
              if (extension != null) {
                extension = extension.ToLower();
              }
              if (extension != ".gbpl") {
                Console.WriteLine("GPUVerify: error: {0} is not a .gbpl file", file);
                Environment.Exit(1);
              }
            }

            parseProcessOutput();
          } catch (Exception e) {
            Console.Error.WriteLine("Exception thrown in GPUVerifyVCGen");
            Console.Error.WriteLine(e);

            if(GPUVerifyVCGenCommandLineOptions.DebugGPUVerify) {
              throw e;
            }
            
            Environment.Exit(1);
          }
        }

        public static Program parse(out ResolutionContext rc)
        {
            Program program = ParseBoogieProgram(GPUVerifyVCGenCommandLineOptions.inputFiles, false);
            if (program == null)
            {
                Environment.Exit(1);
            }

            Microsoft.Boogie.CommandLineOptions.Clo.DoModSetAnalysis = true;

            rc = new ResolutionContext(null);
            program.Resolve(rc);
            if (rc.ErrorCount != 0)
            {
                Console.WriteLine("{0} name resolution errors detected in {1}", rc.ErrorCount, GPUVerifyVCGenCommandLineOptions.inputFiles[GPUVerifyVCGenCommandLineOptions.inputFiles.Count - 1]);
                Environment.Exit(1);
            }
            
            int errorCount = program.Typecheck();
            if (errorCount != 0)
            {
                Console.WriteLine("{0} type checking errors detected in {1}", errorCount, GPUVerifyVCGenCommandLineOptions.inputFiles[GPUVerifyVCGenCommandLineOptions.inputFiles.Count - 1]);
                Environment.Exit(1);
            }

            return program;
        }

        private static Variable findClonedVar(Variable v1, ICollection<Variable> vars)
        {
            foreach (Variable v2 in vars)
            {
                if (v1.Name.Equals(v2.Name))
                {
                    return v2;
                }
            }
            return null;
        }

        public static void parseProcessOutput()
        {
            string fn = "temp";
            if (GPUVerifyVCGenCommandLineOptions.outputFile != null)
            {
                fn = GPUVerifyVCGenCommandLineOptions.outputFile;
            }
            else if (GPUVerifyVCGenCommandLineOptions.inputFiles.Count == 1)
            {
                var inputFile = GPUVerifyVCGenCommandLineOptions.inputFiles[0];
                if (Path.GetExtension(inputFile).ToLower() != ".bpl")
                    fn = Path.GetFileNameWithoutExtension(inputFile);
            }
            ResolutionContext rc;
            Program program = parse(out rc);
            new GPUVerifier(fn, program, rc, GPUVerifyVCGenCommandLineOptions.size_t_bits).doit();
        }

        public static Program ParseBoogieProgram(List<string> fileNames, bool suppressTraceOutput)
        {
            Microsoft.Boogie.CommandLineOptions.Install(new Microsoft.Boogie.CommandLineOptions());

            Program program = null;
            bool okay = true;
            for (int fileId = 0; fileId < fileNames.Count; fileId++)
            {
                string bplFileName = fileNames[fileId];

                Program programSnippet;
                int errorCount;
                try
                {
                    var defines = new List<string>() { "FILE_" + fileId };
                    errorCount = Parser.Parse(bplFileName, defines, out programSnippet);
                    if (programSnippet == null || errorCount != 0)
                    {
                        Console.WriteLine("{0} parse errors detected in {1}", errorCount, bplFileName);
                        okay = false;
                        continue;
                    }
                }
                catch (IOException e)
                {
                    Console.WriteLine("GPUVerify: error opening file \"{0}\": {1}", bplFileName, e.Message);
                    okay = false;
                    continue;
                }
                if (program == null)
                {
                    program = programSnippet;
                }
                else if (programSnippet != null)
                {
                    program.TopLevelDeclarations.AddRange(programSnippet.TopLevelDeclarations);
                }
            }
            if (!okay)
            {
                return null;
            }
            else if (program == null)
            {
                return new Program();
            }
            else
            {
                return program;
            }
        }
    }
}
