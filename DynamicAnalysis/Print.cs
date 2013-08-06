using System;

namespace DynamicAnalysis
{
	public class Print
	{
		public static short debug = 0;
		public static bool verbose = false;
		
		public static void DebugMessage (string arg, short level)
		{
			if (level >= debug)
				Console.WriteLine(arg);
		}
		
		public static void VerboseMessage (string arg)
		{
			if (verbose)
				Console.WriteLine(arg);
		}
		
		public static void ExitMessage (string arg)
		{
			Console.WriteLine("ERROR: {0}", arg);
			Environment.Exit(1);
		}
	}
}

