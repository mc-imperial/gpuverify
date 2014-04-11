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
    
    // Assume a sequential pipeline unless the user selects otherwise
    public Pipeline Pipeline = new Pipeline(sequential: true);

    public GPUVerifyCruncherCommandLineOptions() :
      base() { }

    protected override bool ParseOption(string name, CommandLineOptionEngine.CommandLineParseState ps)
    {
      if (name == "sequential") {
        if (ps.ConfirmArgumentCount(1)) { 
          ParsePipelineString(ps.args[ps.i]);
        }
        return true;
      }
      
      if (name == "parallel") {
        if (ps.ConfirmArgumentCount(1)) {
          Pipeline.Sequential = false;
          ParsePipelineString(ps.args[ps.i]);
        }
        return true;
      }
      
      if (name == "delayHoudini") {
        if (ps.ConfirmArgumentCount(1))
        {
          int houdiniDelay = 0;
          if (ps.GetNumericArgument(ref houdiniDelay))
            Pipeline.houdiniDelay = houdiniDelay;
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
    
    private void ParsePipelineString (string pipelineStr)
	{
      Debug.Assert (pipelineStr[0] == '[' && pipelineStr[pipelineStr.Length - 1] == ']');
	  string[] engines = pipelineStr.Substring(1, pipelineStr.Length - 2).Split('-');
	  foreach (string engine in engines) 
      {
        if (engine.ToUpper().StartsWith("HOUDINI")) 
        {
          Dictionary<string, string> parameters = GetParameters(engine.Substring(2));
          Pipeline.AddEngine(new VanillaHoudini(Pipeline.GetNextSMTEngineID(), 
                                                GetSolverValue(parameters),
                                                GetErrorLimitValue(parameters)));
        }
        else if (engine.ToUpper().StartsWith("SBASE")) 
        {
          Dictionary<string, string> parameters = GetParameters(engine.Substring(2));
          Pipeline.AddEngine(new SBASE(Pipeline.GetNextSMTEngineID(), 
                                       GetSolverValue(parameters), 
                                       GetErrorLimitValue(parameters)));
		}
		else if (engine.ToUpper().StartsWith("SSTEP"))
        {
          Dictionary<string, string> parameters = GetParameters(engine.Substring(2));
          Pipeline.AddEngine(new SSTEP(Pipeline.GetNextSMTEngineID(), 
                                       GetSolverValue(parameters), 
                                       GetErrorLimitValue(parameters)));
        }
        else if (engine.ToUpper().StartsWith("LU"))
        {
          Dictionary<string, string> parameters = GetParameters(engine.Substring(2));
          if (!parameters.ContainsKey(LU.UnrollParam))
          {  
            Console.WriteLine(String.Format("For LU you must supply the parameter '{0}'", LU.UnrollParam));
            System.Environment.Exit(1);
          }
          Pipeline.AddEngine(new LU(Pipeline.GetNextSMTEngineID(), 
                                    ParseIntParameter(LU.UnrollParam, parameters[LU.UnrollParam]), 
                                    GetSolverValue(parameters), 
                                    GetErrorLimitValue(parameters)));
        }
        else if (engine.ToUpper().Equals("DYNAMIC"))
        {
            Pipeline.AddEngine(new DynamicAnalysis());
        }
        else
        {
            Console.WriteLine(String.Format("Unknown cruncher engine: '{0}'", engine));  
            System.Environment.Exit(1);
        }
      }
    } 
    
    private Dictionary<string, string> GetParameters(string parameterStr)
    {
      Dictionary<string, string> map = new Dictionary<string, string>();
      Debug.Assert (parameterStr[0] == '[' && parameterStr[parameterStr.Length - 1] == ']');
      string[] parameters = parameterStr.Substring(1, parameterStr.Length - 2).Split(',');
      foreach (string param in parameters)
      {
        string[] values = param.Split('=');
        Debug.Assert(values.Length == 2);
        map[values[0]] = values[1].ToLower();
      }
      return map;
    }
    
    private int GetErrorLimitValue(Dictionary<string, string> parameters)
    {
      if (parameters.ContainsKey(SMTEngine.ErrorLimitParam))
        return ParseIntParameter(SMTEngine.ErrorLimitParam, parameters[SMTEngine.ErrorLimitParam]);
      return SMTEngine.DefaultErrorLimit;
    }
    
    private string GetSolverValue(Dictionary<string, string> parameters)
    {
      if (parameters.ContainsKey(SMTEngine.SolverParam))
      {
        if (!parameters[SMTEngine.SolverParam].Equals(SMTEngine.Z3) && !parameters[SMTEngine.SolverParam].Equals(SMTEngine.CVC4))
        {
          Console.WriteLine(String.Format("Unknown solver '{0}'", parameters[SMTEngine.SolverParam]));
          System.Environment.Exit(1);
        }
        return parameters[SMTEngine.SolverParam];
      }
      return SMTEngine.DefaultSolver;
    }
    
    private int ParseIntParameter(string paramName, string paramValue)
    {
      try
      {
        return Convert.ToInt32(paramValue);
      }            
      catch (FormatException)
      {
        Console.WriteLine(String.Format("'{0}' must be an integer. You gave '{1}'", paramName, paramValue));
        System.Environment.Exit(1);
      }
      catch (OverflowException)
      {
        Console.WriteLine(String.Format("'{0}' must fit into a 32-bit integer. You gave '{1}'", paramName, paramValue));
        System.Environment.Exit(1);
      }
      return -1;
    }
  }   
}
