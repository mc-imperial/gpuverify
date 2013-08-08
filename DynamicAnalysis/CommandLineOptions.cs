using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;

namespace DynamicAnalysis
{
	public class CommandLineOptions
	{
		private static bool Help = false;
		private static List<string> Files = new List<string>();
		
		public static void Parse(string[] args)
		{	
			for (int i = 0; i < args.Length; i++)
            {
                string beforeColon;
                string afterColon = null;
                int colonIndex = args[i].IndexOf(':');
                if (colonIndex >= 0 && (args[i].StartsWith("-") || args[i].StartsWith("/"))) 
				{
                    beforeColon = args[i].Substring(0, colonIndex);
                    afterColon = args[i].Substring(colonIndex + 1);
                } 
				else 
				{
                    beforeColon = args[i];
                }
				
				switch (beforeColon)
				{
				case "-h":
                case "/h":
				case "-help":
                case "/help":
                case "-?":
                case "/?":
					Help = true;
					break;
					
				case "-d":
				case "/d":
					HandleDebug(afterColon);
					break;
					
				case "-v":
				case "/v":
					Print.verbose = true;
					break;
					
				default:
					Files.Add(args[i]);
					break;
				}
			}
			
			if (Help)
				CommandLineOptions.Usage();
			else
				HandleFiles();			
		}
		
		public static string GetBoogieFile ()
		{
			return Files[0];
		}
		
		private static void HandleDebug (string val)
		{
			int intVal = -1;
			try
			{
				intVal = Convert.ToInt32(val);
			}
			catch (FormatException)
	        {
	            Print.ExitMessage(String.Format("Debug value '{0}' is not an integer", val));
	        }
	        catch (OverflowException)
	        {
	            Print.ExitMessage(String.Format("The number '{0}' cannot fit as a 16-bit integer", val));
	        }
			finally 
			{
				if (intVal < 0)
					Print.ExitMessage("Debug values must be non-negative integers");
			}
			Print.debug = intVal;
		}
		
		private static void HandleFiles ()
		{
			Debug.Assert(Files.Count == 1, "You must only pass a single file");		
			string file = Files[0];
			string ext  = Path.GetExtension(file);
			if (ext != null) 
			{
                ext = ext.ToLower();
            }
			Debug.Assert(ext == ".bpl" || ext == ".gbpl", String.Format("'{0}' is not a .bpl or .gbpl file", file));
			Debug.Assert(File.Exists(file), String.Format("File '{0}' does not exist", file));
		}
		
		public static void Usage()
        {
			Console.WriteLine(@"DynamicAnalysis: usage:  DynamicAnalysis [ option ... ] [ filename ... ]
  where <option> is one of

  /h                            : output this message
  /d:level                      : output debug messages at or above the specified level
  /v                            : output verbose messages
");
			Environment.Exit(0);
        }

	}
}

