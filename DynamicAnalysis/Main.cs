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
			string file = CommandLineOptions.getBoogieFile();
			Program program;
			var defines = new List<string>() { "FILE_0" };
			int errors = Microsoft.Boogie.Parser.Parse(file, defines, out program);
			Debug.Assert(errors == 0, String.Format("Parse errors discovered in '{0}'", file));
			BoogieInterpreter.interpret(program);
		}
	}
}
