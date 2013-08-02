using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;

namespace DynamicAnalysis
{
	public class CommandLineOptions
	{
		private static Dictionary<string, object> values = new Dictionary<string, object>();
		
		public static void Parse(string[] args)
		{	
			values["help"]    = false;
			values["verbose"] = false;
			values["debug"]   = "0";
			values["files"]   = new List<string>();
			
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
					values["help"] = true;
					break;
					
				case "-d":
				case "/d":
					values["debug"] = afterColon;
					break;
					
				case "-v":
				case "/v":
					values["verbose"] = true;
					break;
					
				default:
					((List<string>) values["files"]).Add(args[i]);
					break;
				}
			}
			
			if ((bool) values["help"])
				CommandLineOptions.Usage();
			
			Print.verbose = ((bool) values["verbose"]);
			handleDebug();
			handleFiles();			
		}
		
		public static string getBoogieFile ()
		{
			List<string> files = (List<string>) values["files"];
			return files[0];
		}
		
		private static void handleDebug ()
		{
			string val   = (string) values["debug"];
			short intVal = -1;
			try
			{
				intVal = Convert.ToInt16(val);
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
		
		private static void handleFiles ()
		{
			List<string> files = (List<string>) values["files"];
			Debug.Assert(files.Count == 1, "You must pass a single .bpl file");		
			string file = files[0];
			string ext  = Path.GetExtension(file);
			if (ext != null) 
			{
                ext = ext.ToLower();
            }
			Debug.Assert(ext == ".bpl", String.Format("'{0}' is not a .bpl file", file));
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

