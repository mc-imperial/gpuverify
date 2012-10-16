using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;

namespace GPUVerify
{

    class CommandLineOptions
    {

        public static List<string> inputFiles = new List<string>();

        public static string outputFile = null;

        public static bool OnlyDivergence = false;
        public static bool AdversarialAbstraction = false;
        public static bool EqualityAbstraction = false;
        public static bool Inference = false;
        public static bool ArrayEqualities = false;
        public static string invariantsFile = null;

        public static bool ShowStages = false;

        public static bool ShowUniformityAnalysis = false;
        public static bool DoUniformityAnalysis = true;

        public static bool ShowMayBePowerOfTwoAnalysis = false;
        public static bool ShowArrayControlFlowAnalysis = false;

        public static bool NoLoopPredicateInvariants = false;

        public static bool Unstructured = true;
        public static bool SmartPredication = true;

        public static bool OnlyIntraGroupRaceChecking = false;

        public static bool InferSourceLocation = true;

        public static int Parse(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                bool hasColonArgument = false;
                string beforeColon;
                string afterColon = null;
                int colonIndex = args[i].IndexOf(':');
                if (colonIndex >= 0 && (args[i].StartsWith("-") || args[i].StartsWith("/"))) {
                    hasColonArgument = true;
                    beforeColon = args[i].Substring(0, colonIndex);
                    afterColon = args[i].Substring(colonIndex + 1);
                } else {
                    beforeColon = args[i];
                }

                switch (beforeColon)
                {
                    case "-help":
                    case "/help":
                    case "-?":
                    case "/?":
                    return -1;
                    
                    case "-print":
                    case "/print":
                        if (!hasColonArgument)
                        {
                            Console.WriteLine("Error: filename expected after " + beforeColon + " argument");
                            Environment.Exit(1);
                        }
                        Debug.Assert(afterColon != null);
                        outputFile = afterColon;
                    break;

                    case "-onlyDivergence":
                    case "/onlyDivergence":
                    OnlyDivergence = true;

                    break;

                    case "-adversarialAbstraction":
                    case "/adversarialAbstraction":
                    AdversarialAbstraction = true;

                    break;

                    case "-equalityAbstraction":
                    case "/equalityAbstraction":
                    EqualityAbstraction = true;

                    break;

                    case "-showStages":
                    case "/showStages":
                    ShowStages = true;
                    break;

                    case "-inference":
                    case "/inference":
                    Inference = true;
                    if (hasColonArgument)
                    {
                        Debug.Assert(afterColon != null);
                        invariantsFile = afterColon;
                    }

                    break;

                    case "-arrayEqualities":
                    case "/arrayEqualities":
                    ArrayEqualities = true;
                    break;

                    case "-showUniformityAnalysis":
                    case "/showUniformityAnalysis":
                    ShowUniformityAnalysis = true;
                    break;

                    case "-noUniformityAnalysis":
                    case "/noUniformityAnalysis":
                    DoUniformityAnalysis = false;
                    break;

                    case "-showMayBePowerOfTwoAnalysis":
                    case "/showMayBePowerOfTwoAnalysis":
                    ShowMayBePowerOfTwoAnalysis = true;
                    break;

                    case "-showArrayControlFlowAnalysis":
                    case "/showArrayControlFlowAnalysis":
                    ShowArrayControlFlowAnalysis = true;
                    break;

                    case "-noLoopPredicateInvariants":
                    case "/noLoopPredicateInvariants":
                    NoLoopPredicateInvariants = true;
                    break;

                    case "-structured":
                    case "/structured":
                    Unstructured = false;
                    break;

                    case "-noSmartPredication":
                    case "/noSmartPredication":
                    SmartPredication = false;
                    break;

                    case "-onlyIntraGroupRaceChecking":
                    case "/onlyIntraGroupRaceChecking":
                    OnlyIntraGroupRaceChecking = true;
                    break;

                    case "-noSourceLocInfer":
                    case "/noSourceLocInfer":
                    InferSourceLocation = false;
                    break;

                    default:
                        inputFiles.Add(args[i]);
                        break;
                }

            }
            return 0;
        }

        private static bool printedHelp = false;

        public static void Usage()
        {
            // Ensure that we only print the help message once
            if (printedHelp)
            {
                return;
            }
            printedHelp = true;

            Console.WriteLine(@"GPUVerify: usage:  GPUVerify [ option ... ] [ filename ... ]
  where <option> is one of

  /help                         : this message
  /print:file                   : output bpl file
  /showStages                   :

  /adversarialAbstraction       : apply full state abstraction
  /equalityAbstraction          : apply equality state abstraction

  /inference[:file]             : use automatic invariant inference
                                  optional file can include manually supplied invariants

  /onlyDivergence               : only-check for divergence-freedom, not race-freedom
  /onlyIntraGroupRaceChecking   : only-check intra-group races

  /arrayEqualities              :
  /noLoopPredicateInvariants    :
  /noSmartPredication           :
  /noUniformityAnalysis         :
  /noSourceLocInfer             : turn off source-location tags
  /showArrayControlFlowAnalysis :
  /showMayBePowerOfTwoAnalysis  :
  /showUniformityAnalysis       :
  /structured                   : work on structured form of program (default: unstructured)

");
        }


    }
}
