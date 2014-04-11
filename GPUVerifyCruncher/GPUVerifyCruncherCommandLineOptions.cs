//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

using System.Collections.Generic;
using System;
using System.Linq;
using System.Text;
using System.Diagnostics;
using Microsoft.Boogie;

namespace GPUVerify
{
  public class GPUVerifyCruncherCommandLineOptions : GVCommandLineOptions
  {
    public string ConfigFile = "inference.cfg";
    public string ParallelInferenceScheduling = "default";
    public string RefutationEngine = "";
    public int InferenceSliding = 0;
    public int InferenceSlidingLimit = 1;
    public int DynamicErrorLimit = 0;
    public int DelayHoudini = 0;
    public bool ParallelInference = false;
    public bool DynamicAnalysis = false;
    public bool InferInfo = false;
    public int DynamicAnalysisLoopHeaderLimit = 1000;
    public int DynamicAnalysisUnsoundLoopEscaping = 0;
    public bool DynamicAnalysisSoundLoopEscaping = false;
    public bool ReplaceLoopInvariantAssertions = false;
    public bool EnableBarrierDivergenceChecks = false;
    
    public List<Engine> SequentialPipeline;
    public HashSet<Engine> ParallelPipeline;

    public GPUVerifyCruncherCommandLineOptions() :
      base() { }

    protected override bool ParseOption(string name, CommandLineOptionEngine.CommandLineParseState ps)
    {
      if (name == "sequentialCrunch") {
        if (ps.ConfirmArgumentCount(1)) {
          SequentialPipeline = ParsePipeline(ps.args[ps.i]);
        }
        return true;
      }
      
      if (name == "parallelCrunch") {
        if (ps.ConfirmArgumentCount(1)) {
          ParallelPipeline = new HashSet<Engine>(ParsePipeline(ps.args[ps.i]));
        }
        return true;
      }
    
      if (name == "invInferConfigFile") {
        if (ps.ConfirmArgumentCount(1)) {
          ConfigFile = ps.args[ps.i];
        }
        return true;
      }

      if (name == "parallelInferenceScheduling") {
        if (ps.ConfirmArgumentCount(1)) {
          ParallelInferenceScheduling = ps.args[ps.i];
        }
        return true;
      }

      if (name == "refutationEngine") {
        if (ps.ConfirmArgumentCount(1)) {
           RefutationEngine = ps.args[ps.i];
        }
        return true;
      }

      if (name == "inferenceSliding") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref InferenceSliding);
        return true;
      }

      if (name == "inferenceSlidingLimit") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref InferenceSlidingLimit);
        return true;
      }

      if (name == "parallelInference") {
        ParallelInference = true;
        return true;
      }

      if (name == "dynamicErrorLimit") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref DynamicErrorLimit);
        return true;
      }

      if (name == "delayHoudini") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref DelayHoudini);
        return true;
      }

      if (name == "dynamicAnalysis") {
        DynamicAnalysis = true;
        return true;
      }
      
      if (name == "dynamicAnalysisSoundLoopEscaping") {
        DynamicAnalysisSoundLoopEscaping = true;
        return true;
      }
      
      if (name == "dynamicAnalysisUnsoundLoopEscaping") {
         if (ps.ConfirmArgumentCount(1))
           ps.GetNumericArgument(ref DynamicAnalysisUnsoundLoopEscaping);
         return true;
      }

      if (name == "dynamicAnalysisLoopHeaderLimit") {
         if (ps.ConfirmArgumentCount(1))
           ps.GetNumericArgument(ref DynamicAnalysisLoopHeaderLimit);
         return true;
      }

      if (name == "inferInfo") {
        InferInfo = true;
        return true;
      }

      if (name == "replaceLoopInvariantAssertions") {
        ReplaceLoopInvariantAssertions = true;
        return true;
      }

      if (name == "enableBarrierDivergenceChecks") {
        EnableBarrierDivergenceChecks = true;
        return true;
      }

      return base.ParseOption(name, ps);  // defer to superclass
    }
    
    private List<Engine> ParsePipeline (string pipeline)
		{
			int SMTBasedID = 0;
			List<Engine> engineList = new List<Engine> ();
			Debug.Assert (pipeline [0] == '[' && pipeline [pipeline.Length - 1] == ']');
			string[] engines = pipeline.Substring (1, pipeline.Length - 2).Split (',');
			foreach (string engine in engines) 
      {
        if (engine.ToUpper().Equals("HOUDINI"))
        {
          engineList.Add(new ClassicHoudini(SMTBasedID));
          ++SMTBasedID;
        }
				else if (engine.ToUpper().Equals ("SBASE")) 
        {
					engineList.Add (new SBASE (SMTBasedID));
          ++SMTBasedID;
				}
				else if (engine.ToUpper().Equals("SSTEP"))
        {
          engineList.Add(new SSTEP(SMTBasedID));
          ++SMTBasedID;
        }
        else if (engine.ToUpper().StartsWith("LU"))
        {
            try
            {
              int unrollFactor = Convert.ToInt32(engine.Substring(2));
              engineList.Add(new LU(SMTBasedID, unrollFactor));
              ++SMTBasedID;
            }
            catch (FormatException)
            {
                Console.WriteLine("Loop unroll factor must be a number. You gave: " + engine);
                System.Environment.Exit(1);
            }
            catch (OverflowException)
            {
                Console.WriteLine("Loop unroll factor must fit into a 32-bit integer. You gave: " + engine);
                System.Environment.Exit(1);
            }
        }
        else if (engine.ToUpper().Equals("DYNAMIC"))
        {
            engineList.Add(new DynamicAnalysis());
        }
        else
        {
            Console.WriteLine(String.Format("Unknown cruncher engine: '{0}'", engine));  
            System.Environment.Exit(1);
        }
      }
      return engineList;
    } 
  }   
}
