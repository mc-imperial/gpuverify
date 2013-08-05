using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Boogie;

namespace DynamicAnalysis
{
	class MainClass
	{
		public static void Main (string[] args)
		{
			CommandLineOptions.Parse(args);
			Microsoft.Boogie.CommandLineOptions.Install(new OverrideCommandLineOptions());
			string file = CommandLineOptions.GetBoogieFile();
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
			BoogieInterpreter.Interpret(program);
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
