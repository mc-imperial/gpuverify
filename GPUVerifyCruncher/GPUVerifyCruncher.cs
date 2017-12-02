//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.Contracts;
    using System.IO;
    using Microsoft.Boogie;

    public class GPUVerifyCruncher
    {
        public static void Main(string[] args)
        {
            Contract.Requires(cce.NonNullElements(args));
            CommandLineOptions.Install(new GPUVerifyCruncherCommandLineOptions());

            try
            {
                CommandLineOptions.Clo.RunningBoogieFromCommandLine = true;

                if (!CommandLineOptions.Clo.Parse(args))
                {
                    Environment.Exit((int)ToolExitCodes.OTHER_ERROR);
                }

              ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParsePipelineString();

                if (CommandLineOptions.Clo.Files.Count == 0)
                {
                    GVUtil.IO.ErrorWriteLine("GPUVerify: error: no input files were specified");
                    Environment.Exit((int)ToolExitCodes.OTHER_ERROR);
                }

                if (!CommandLineOptions.Clo.DontShowLogo)
                {
                    Console.WriteLine(CommandLineOptions.Clo.Version);
                }

                List<string> fileList = new List<string>();

                foreach (string file in CommandLineOptions.Clo.Files)
                {
                    string extension = Path.GetExtension(file)?.ToLower();

                    fileList.Add(file);
                }

                foreach (string file in fileList)
                {
                    Contract.Assert(file != null);
                    string extension = Path.GetExtension(file)?.ToLower();

                    if (extension != ".bpl")
                    {
                        GVUtil.IO.ErrorWriteLine("GPUVerify: error: {0} is not a .bpl file", file);
                        Environment.Exit((int)ToolExitCodes.OTHER_ERROR);
                    }
                }

                Scheduler scheduler = new Scheduler(fileList);
                Environment.Exit(scheduler.ErrorCode);
            }
            catch (Exception e)
            {
                if (GetCommandLineOptions().DebugGPUVerify)
                {
                    Console.Error.WriteLine("Exception thrown in GPUVerifyCruncher");
                    Console.Error.WriteLine(e);
                    throw e;
                }

                GVUtil.IO.DumpExceptionInformation(e);

                Environment.Exit((int)ToolExitCodes.INTERNAL_ERROR);
            }
        }

        private static GPUVerifyCruncherCommandLineOptions GetCommandLineOptions()
        {
            return (GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo;
        }
    }
}
