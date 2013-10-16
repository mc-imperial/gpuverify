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
using System.Diagnostics;
using Microsoft.Boogie;

namespace DynamicAnalysis
{
	public class MainClass
	{
		public static void Main (string[] args)
		{
			CommandLineOptions.Parse(args);
			Microsoft.Boogie.CommandLineOptions.Install(new OverrideCommandLineOptions());
			string file = CommandLineOptions.Files[0];
			Program program;
			var defines = new List<string>() { "FILE_0" };
			int errors = Microsoft.Boogie.Parser.Parse(file, defines, out program);
			Debug.Assert(errors == 0, String.Format("Parse errors discovered in '{0}'", file));
			errors = program.Resolve();
			errors = program.Typecheck();
			Microsoft.Boogie.CommandLineOptions.Clo.PrintUnstructured = 2;
			using (TokenTextWriter writer = new TokenTextWriter(file + ".unstructured"))
			{
        		writer.WriteLine();
        		program.Emit(writer);
			}
			Start(program, CommandLineOptions.ThreadID, CommandLineOptions.GroupID, Print.verbose, Print.debug);
		}
		
		public static void Start (Program program, Tuple<int, int, int> threadID, Tuple<int, int, int> groupID, bool verbose = false, int debug = 0)
		{
			Print.verbose = verbose;
			Print.debug = debug;
			new BoogieInterpreter(program, threadID, groupID);
		}
	}
	
	public class OverrideCommandLineOptions : Microsoft.Boogie.CommandLineOptions 
	{
		public OverrideCommandLineOptions() :
		base("Dynamic", "Dynamic analysis of Boogie code") 
		{
		}
	}
}
