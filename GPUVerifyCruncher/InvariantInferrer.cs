//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


﻿using System;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;
using Microsoft.Boogie;

namespace GPUVerify
{
  public class InvariantInferrer
  {
    Configuration config = null;
    RefutationEngine[] refutationEngines = null;
    int numRefEng = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).NumOfRefutationEngines;

    public InvariantInferrer()
    {
      refutationEngines = new RefutationEngine[numRefEng];
      config = new Configuration();
    }

    public int inferInvariants(Program program)
    {
      int exitCode = 0;
      string conf;
      string engine;

      for (int i = 0; i < numRefEng; i++) {
        if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
          engine = "Engine_" + (i + 1);
          conf = config.getValue("ParallelInference", engine);
        } else {
          conf = config.getValue("Inference", "Engine");
        }

        refutationEngines[i] = new RefutationEngine(0, conf,
                                                    config.getValue(conf, "Solver"),
                                                    config.getValue(conf, "ErrorLimit"),
                                                    config.getValue(conf, "CheckForLMI"),
                                                    config.getValue(conf, "ModifyTSO"),
                                                    config.getValue(conf, "LoopUnwind"));
        exitCode += refutationEngines[i].start(program);
      }

      return exitCode;
    }

    public void applyInvariants(Program program)
    {
      if (((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ParallelInference) {
        // do nothing currently
      }

      if (refutationEngines != null && refutationEngines[0] != null) {
        refutationEngines[0].houdini.ApplyAssignment(program);
      }
    }

    private class Configuration
    {
      Dictionary<string, Dictionary<string, string>> info = null;

      public Configuration()
      {
        info = new Dictionary<string, Dictionary<string, string>>();
        updateFromConfigurationFile();
      }

      public string getValue(string key1, string key2)
      {
        Console.WriteLine(key1 + " :: " + key2 + " :: " + info[key1][key2]);
        return info[key1][key2];
      }

      private void updateFromConfigurationFile()
      {
        string file = ((GPUVerifyCruncherCommandLineOptions)CommandLineOptions.Clo).ConfigFile;

        try {
          using (var fileStream = new FileStream(file, FileMode.Open, FileAccess.Read))
          using (var input = new StreamReader(fileStream)) {
            string entry;
            string key = "";

            while ((entry = input.ReadLine()) != null) {
              entry = Regex.Replace(entry, ";.*", "");
              if (entry.Length == 0) continue;
              if (entry.StartsWith("[")) {
                key = Regex.Replace(entry, "[[\\]]+", "");
                info.Add(key, new Dictionary<string, string>());
              }
              else {
                if (key.Length == 0) throw new Exception();
                string[] tokens = new Regex("[ =\t]+").Split(entry);
                if (tokens.Length != 2) throw new Exception();
                info[key].Add(tokens[0], tokens[1]);
              }
            }
          }
        } catch (FileNotFoundException e) {
          Console.Error.WriteLine("{0}: The configuration file {1} was not found", e.GetType(), file);
          Environment.Exit(1);
        } catch (Exception e) {
          Console.Error.WriteLine("{0}: The file {1} is not properly formatted", e.GetType(), file);
          Environment.Exit(1);
        }
      }

//      public void print()
//      {
//        Console.WriteLine(### Configuration Options for Invariant Inference ###");
//
//        foreach (KeyValuePair<string, Dictionary<string, string>> kvpExt in info) {
//          Console.Write("Option: " + kvpExt.Key);
//
//          foreach (KeyValuePair<string, string> kvpInt in kvpExt) {
//            Console.Write(" :: " + kvpInt.Key + " :: " + kvpInt.Value);
//          }
//
//          Console.WriteLine();
//        }
//      }
    }
  }
}
