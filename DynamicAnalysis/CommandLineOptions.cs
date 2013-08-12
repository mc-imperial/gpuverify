using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Text.RegularExpressions;
using Microsoft.Boogie;

namespace DynamicAnalysis
{
	public class CommandLineOptions
	{
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
					CommandLineOptions.Usage();
					break;
					
				case "-d":
				case "/d":
					HandleDebug(afterColon);
					break;
					
				case "-v":
				case "/v":
					Print.verbose = true;
					break;
					
				case "-tid":
				case "/tid":
					Tuple<int, int, int> threadID = HandleTuple(afterColon);
					GPU.Instance.SetThreadID(threadID);
					break;
					
				case "-gid":
				case "/gid":
					Tuple<int, int, int> groupID = HandleTuple(afterColon);
					GPU.Instance.SetGroupID(groupID);
					break;
					
				case "-blockDim":
				case "/blockDim":
					Tuple<int, int, int> blockDim = HandleTuple(afterColon);
					GPU.Instance.SetBlockDim(blockDim);
					break;
					
				case "-gridDim":
				case "/gridDim":
					Tuple<int, int, int> gridDim = HandleTuple(afterColon);
					GPU.Instance.SetGridDim(gridDim);
					break;
					
				default:
					Files.Add(args[i]);
					break;
				}
			}
			// Grab the Boogie files
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
	            Print.ExitMessage(String.Format("The number '{0}' cannot fit as a 32-bit integer", val));
	        }
			finally 
			{
				if (intVal < 0)
					Print.ExitMessage("Debug values must be non-negative integers");
			}
			Print.debug = intVal;
		}
		
		private static Tuple<int, int, int> HandleTuple (string val)
		{
			Print.ConditionalExitMessage(val.StartsWith("]") || val.EndsWith("]"), String.Format("Tuple string '{0}' must begin with '[' and end with ']'", val));
			string newval = val.Replace("[", "").Replace("]", "");
			string[] lexemes = Regex.Split(newval, @","); 
			int x = 1, y = 1, z = 1;
			DIMENSION dim = DIMENSION.X;
			foreach (string lex in lexemes)
			{
				int intVal = -1;
				try
				{
					intVal = Convert.ToInt32(lex);
					switch (dim)
					{
					case DIMENSION.X:
						x = Convert.ToInt32(lex);
						break;
					case DIMENSION.Y:
						y = Convert.ToInt32(lex);
						break;
					case DIMENSION.Z:
						z = Convert.ToInt32(lex);
						break;
					}
				}
				catch (FormatException)
		        {
		            Print.ExitMessage(String.Format("'{0}' is not an integer", lex));
		        }
		        catch (OverflowException)
		        {
		            Print.ExitMessage(String.Format("The number '{0}' cannot fit as a 32-bit integer", lex));
		        }
				finally 
				{
					if (intVal < 0)
						Print.ExitMessage(String.Format("Tuple value '{0}' cannot be a negative integer", intVal));
				}
				dim++;
			}
			return Tuple.Create(x, y, z);
		}
		
		private static void HandleFiles ()
		{
			Print.ConditionalExitMessage(Files.Count == 1, "You must only pass a single file");		
			string file = Files[0];
			string ext  = Path.GetExtension(file);
			if (ext != null) 
			{
                ext = ext.ToLower();
            }
			Print.ConditionalExitMessage(ext == ".bpl" || ext == ".gbpl", String.Format("'{0}' is not a .bpl or .gbpl file", file));
			Print.ConditionalExitMessage(File.Exists(file), String.Format("File '{0}' does not exist", file));
		}
		
		public static void Usage()
        {
			Console.WriteLine(@"DynamicAnalysis: usage:  DynamicAnalysis [ option ... ] [ filename ... ]
  where <option> is one of

  /h                            : output this message
  /d:level                      : output debug messages at or above the specified level
  /v                            : output verbose messages
  /tid:[x,y,z]                  : set the thread ID
  /gid:[x,y,z]                  : set the group ID
  /blockDim:[x,y,z]             : set the block dimensions
  /gridDim:[x,y,z]              : set the grid dimensions
");
			Environment.Exit(0);
        }

	}
}

