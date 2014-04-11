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
    public int DynamicErrorLimit = 0;
    public bool InferInfo = false;
    public int DynamicAnalysisLoopHeaderLimit = 1000;
    public int DynamicAnalysisUnsoundLoopEscaping = 0;
    public bool DynamicAnalysisSoundLoopEscaping = false;
    public bool ReplaceLoopInvariantAssertions = false;
    public bool EnableBarrierDivergenceChecks = false;
    
    // Assume a sequential pipeline unless the user selects otherwise
    public Pipeline Pipeline = new Pipeline(sequential: true);

    public GPUVerifyCruncherCommandLineOptions() :
      base() 
    { 
    }

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

      if (name == "dynamicErrorLimit") {
        if (ps.ConfirmArgumentCount(1))
            ps.GetNumericArgument(ref DynamicErrorLimit);
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

      return base.ParseOption(name, ps);
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
          
          // The user wants to override Houdini settings used in the cruncher
          VanillaHoudini houdiniEngine = new VanillaHoudini(Pipeline.GetNextSMTEngineID(), 
                                                            GetSolverValue(parameters),
                                                            GetErrorLimitValue(parameters));
          Pipeline.AddEngine(houdiniEngine);
          
          if (parameters.ContainsKey(VanillaHoudini.GetDelayParameter().Name))
          {  
            // Delay Houdini by x seconds
            houdiniEngine.Delay = ParseIntParameter(parameters, VanillaHoudini.GetDelayParameter().Name);
          }
          if (parameters.ContainsKey(VanillaHoudini.GetSlidingSecondsParameter().Name))
          {  
            // Spawn a new Houdini engine after x seconds
            houdiniEngine.SlidingSeconds = ParseFloatParameter(parameters, VanillaHoudini.GetSlidingSecondsParameter().Name);
          }
          if (parameters.ContainsKey(VanillaHoudini.GetSlidingLimitParameter().Name))
          {  
            // Spawn new Houdini engines until this x limit has been reached
            houdiniEngine.SlidingLimit = ParseIntParameter(parameters, VanillaHoudini.GetSlidingLimitParameter().Name);
          }
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
          if (!parameters.ContainsKey(LU.GetUnrollParameter().Name))
          {  
            Console.WriteLine(String.Format("For LU you must supply the parameter '{0}'", LU.GetUnrollParameter().Name));
            System.Environment.Exit(1);
          }
          Pipeline.AddEngine(new LU(Pipeline.GetNextSMTEngineID(), 
                                    GetSolverValue(parameters), 
                                    GetErrorLimitValue(parameters),
                                    ParseIntParameter(parameters, LU.GetUnrollParameter().Name)));
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
      if (parameters.ContainsKey(SMTEngine.GetErrorLimitParameter().Name))
        return ParseIntParameter(parameters, SMTEngine.GetErrorLimitParameter().Name);
      return SMTEngine.GetErrorLimitParameter().DefaultValue;
    }
    
    private string GetSolverValue(Dictionary<string, string> parameters)
    {
      if (parameters.ContainsKey(SMTEngine.GetSolverParameter().Name))
      {
        if (!SMTEngine.GetSolverParameter().IsValidValue(parameters[SMTEngine.GetSolverParameter().Name]))
        {
          Console.WriteLine(String.Format("Unknown solver '{0}'", parameters[SMTEngine.GetSolverParameter().Name]));
          System.Environment.Exit(1);
        }
        return parameters[SMTEngine.GetSolverParameter().Name];
      }
      return SMTEngine.GetSolverParameter().DefaultValue;
    }
    
    private int ParseIntParameter(Dictionary<string, string> parameters, string paramName)
    {
      try
      {
        return Convert.ToInt32(parameters[paramName]);
      }            
      catch (FormatException)
      {
        Console.WriteLine(String.Format("'{0}' must be an integer. You gave '{1}'", paramName, parameters[paramName]));
        System.Environment.Exit(1);
      }
      catch (OverflowException)
      {
        Console.WriteLine(String.Format("'{0}' must fit into a 32-bit integer. You gave '{1}'", paramName, parameters[paramName]));
        System.Environment.Exit(1);
      }
      return -1;
    }
    
    private float ParseFloatParameter(Dictionary<string, string> parameters, string paramName)
    {
      try
      {
        return Convert.ToSingle(parameters[paramName]);
      }            
      catch (FormatException)
      {
        Console.WriteLine(String.Format("'{0}' must be a float. You gave '{1}'", paramName, parameters[paramName]));
        System.Environment.Exit(1);
      }
      catch (OverflowException)
      {
        Console.WriteLine(String.Format("'{0}' must fit into a 32-bit float. You gave '{1}'", paramName, parameters[paramName]));
        System.Environment.Exit(1);
      }
      return -1;
    }
  }   
}
