//===-----------------------------------------------------------------------==//
//
//                GPUVerify - a Verifier for GPU Kernels
//
// This file is distributed under the Microsoft Public License.  See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace GPUVerify
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.Contracts;
    using System.Linq;
    using System.Text;
    using Microsoft.Boogie;
    using VC;

    public static class KernelAnalyser
    {
        public enum PipelineOutcome
        {
            Done,
            ResolutionError,
            TypeCheckingError,
            ResolvedAndTypeChecked,
            FatalError,
            VerificationCompleted
        }

        /// <summary>
        /// Resolves and type checks the given Boogie program.  Any errors are reported to the
        /// console.  Returns:
        ///  - Done if no errors occurred, and command line specified no resolution or no type checking.
        ///  - ResolutionError if a resolution error occurred
        ///  - TypeCheckingError if a type checking error occurred
        ///  - ResolvedAndTypeChecked if both resolution and type checking succeeded
        /// </summary>
        public static PipelineOutcome ResolveAndTypecheck(Program program, string bplFileName)
        {
            Contract.Requires(program != null);
            Contract.Requires(bplFileName != null);

            // ---------- Resolve ----------

            if (CommandLineOptions.Clo.NoResolve)
            {
                return PipelineOutcome.Done;
            }

            int errorCount = program.Resolve();
            if (errorCount != 0)
            {
                Console.WriteLine("{0} name resolution errors detected in {1}", errorCount, bplFileName);
                return PipelineOutcome.ResolutionError;
            }

            // ---------- Type check ----------

            if (CommandLineOptions.Clo.NoTypecheck)
            {
                return PipelineOutcome.Done;
            }

            errorCount = program.Typecheck();
            if (errorCount != 0)
            {
                Console.WriteLine("{0} type checking errors detected in {1}", errorCount, bplFileName);
                return PipelineOutcome.TypeCheckingError;
            }

            if (CommandLineOptions.Clo.PrintFile != null && CommandLineOptions.Clo.PrintDesugarings)
            {
                // if PrintDesugaring option is engaged, print the file here, after resolution and type checking
                GVUtil.IO.PrintBplFile(CommandLineOptions.Clo.PrintFile, program, true);
            }

            return PipelineOutcome.ResolvedAndTypeChecked;
        }

        public static void EliminateDeadVariables(Program program)
        {
            Contract.Requires(program != null);

            // Eliminate dead variables
            Microsoft.Boogie.UnusedVarEliminator.Eliminate(program);

            // Collect mod sets
            if (CommandLineOptions.Clo.DoModSetAnalysis)
            {
                new ModSetCollector().DoModSetAnalysis(program);
            }

            // Coalesce blocks
            if (CommandLineOptions.Clo.CoalesceBlocks)
            {
                if (CommandLineOptions.Clo.Trace)
                    Console.WriteLine("Coalescing blocks...");
                Microsoft.Boogie.BlockCoalescer.CoalesceBlocks(program);
            }
        }

        public static void Inline(Program program)
        {
            Contract.Requires(program != null);

            // Inline
            var TopLevelDeclarations = cce.NonNull(program.TopLevelDeclarations);

            if (CommandLineOptions.Clo.ProcedureInlining != CommandLineOptions.Inlining.None)
            {
                bool inline = false;
                foreach (var d in TopLevelDeclarations)
                {
                    if (d.FindExprAttribute("inline") != null)
                    {
                        inline = true;
                    }
                }

                if (inline)
                {
                    foreach (var d in TopLevelDeclarations)
                    {
                        var impl = d as Implementation;
                        if (impl != null)
                        {
                            impl.OriginalBlocks = impl.Blocks;
                            impl.OriginalLocVars = impl.LocVars;
                        }
                    }

                    foreach (var d in TopLevelDeclarations)
                    {
                        var impl = d as Implementation;
                        if (impl != null && !impl.SkipVerification)
                        {
                            Inliner.ProcessImplementation(program, impl);
                        }
                    }

                    foreach (var d in TopLevelDeclarations)
                    {
                        var impl = d as Implementation;
                        if (impl != null)
                        {
                            impl.OriginalBlocks = null;
                            impl.OriginalLocVars = null;
                        }
                    }
                }
            }
        }

        public static void DisableRaceChecking(Program program)
        {
            foreach (var block in program.Blocks())
            {
                List<Cmd> newCmds = new List<Cmd>();
                foreach (Cmd c in block.Cmds)
                {
                    CallCmd callCmd = c as CallCmd;
                    // TODO: refine into proper check
                    if (callCmd == null || !(callCmd.callee.Contains("_CHECK_READ") ||
                                            callCmd.callee.Contains("_CHECK_WRITE") ||
                                            callCmd.callee.Contains("_CHECK_ATOMIC")))
                    {
                        newCmds.Add(c);
                    }
                }

                block.Cmds = newCmds;
            }
        }

        public static void DisableRaceLogging(Program program)
        {
            foreach (var block in program.Blocks())
            {
                List<Cmd> newCmds = new List<Cmd>();
                foreach (Cmd c in block.Cmds)
                {
                    CallCmd callCmd = c as CallCmd;
                    // TODO: refine into proper check
                    if (callCmd == null || !(callCmd.callee.Contains("_LOG_READ") ||
                                             callCmd.callee.Contains("_LOG_WRITE") ||
                                             callCmd.callee.Contains("_LOG_ATOMIC")))
                    {
                        newCmds.Add(c);
                    }
                }

                block.Cmds = newCmds;
            }
        }

        private static bool IsCandidateAssert(AssertCmd c)
        {
            return QKeyValue.FindStringAttribute(c.Attributes, "tag") != null;
        }

        public static void DisableAssertions(Program program)
        {
            // We want to disable all assertions, with the exception
            // of assertions at loop heads (these are invariants)
            // and candidate assertions that the user has provided
            // for Houdini
            foreach (var impl in program.Implementations)
            {
                var CFG = Program.GraphFromImpl(impl);
                CFG.ComputeLoops();
                foreach (var b in impl.Blocks)
                {
                    var newCmds = new List<Cmd>();
                    bool CmdCouldBeLoopInvariant = CFG.Headers.Contains(b);
                    foreach (var c in b.Cmds)
                    {
                        if (c is AssertCmd)
                        {
                            if (IsCandidateAssert(c as AssertCmd) || CmdCouldBeLoopInvariant)
                            {
                                // Keep it: it's a Houdini candidate or a loop invariant
                                newCmds.Add(c);
                            }
                            else
                            {
                                // Discard the invariant (by not adding it the new list of commands)
                            }
                        }
                        else
                        {
                            newCmds.Add(c);
                        }

                        if (!(c is AssertCmd || c is AssumeCmd))
                        {
                            // On seeing a command that isn't an assertion or an assumption,
                            // we must be past the sequence of assert and assume commands that
                            // form invariants if the block in which they live is a loop header.
                            // Thus future commands in this block cannot be invariants
                            CmdCouldBeLoopInvariant = false;
                        }
                    }

                    b.Cmds = newCmds;
                }
            }
        }

        public static void DisableBarrierDivergenceChecking(Program program)
        {
            foreach (var proc in program.TopLevelDeclarations.OfType<Procedure>())
            {
                List<Requires> newRequires = new List<Requires>();
                foreach (Requires r in proc.Requires)
                {
                    if (!QKeyValue.FindBoolAttribute(r.Attributes, "barrier_divergence"))
                        newRequires.Add(r);
                }

                proc.Requires = newRequires;
            }
        }

        /// <summary>
        /// Checks if Quantifiers exists in the Boogie program. If they exist and the underlying
        /// parser is CVC4 then it enables the corresponding Logic.
        /// </summary>
        public static void CheckForQuantifiersAndSpecifyLogic(Program program, int taskID = -1)
        {
            const string QF_ALL_SUPPORTED = "LOGIC=QF_ALL_SUPPORTED";
            const string ALL_SUPPORTED = "LOGIC=ALL_SUPPORTED";

            // For now it's necessary to handle this separately depending on whether we're using
            // Concurrent Houdini or not.
            if (taskID >= 0)
            {
                var ProverOptions = CommandLineOptions.Clo.Cho[taskID].ProverOptions;
                if (UsingCVC4AndQuantifiersPresent(program, ProverOptions))
                {
                    ProverOptions.Remove(QF_ALL_SUPPORTED);
                    ProverOptions.Add(ALL_SUPPORTED);
                }
            }
            else
            {
                if (UsingCVC4AndQuantifiersPresent(program, CommandLineOptions.Clo.ProverOptions))
                {
                    CommandLineOptions.Clo.ProverOptions = CommandLineOptions.Clo.ProverOptions.Where(Item => !Item.Equals(QF_ALL_SUPPORTED));
                    CommandLineOptions.Clo.ProverOptions = CommandLineOptions.Clo.ProverOptions.Concat1(ALL_SUPPORTED);
                }
            }

        }

        private static bool UsingCVC4AndQuantifiersPresent(Program program, IEnumerable<string> ProverOptions)
        {
            return (ProverOptions.Contains("SOLVER=cvc4") ||
                        ProverOptions.Contains("SOLVER=CVC4")) &&
                        ProverOptions.Contains("LOGIC=QF_ALL_SUPPORTED") &&
                        CheckForQuantifiers.Found(program);
        }

        public static int GetExitCode(ResultCounter counter)
        {
            if (counter.AllVerified())
                return (int)ToolExitCodes.SUCCESS;

            if (counter.HasInternalError())
                return (int)ToolExitCodes.INTERNAL_ERROR;

            if (counter.HasNonInternalError())
                return (int)ToolExitCodes.OTHER_ERROR;

            if (counter.VerificationErrors > 0)
                return (int)ToolExitCodes.VERIFICATION_ERROR;

            // This should be unreachable
            throw new InvalidOperationException("Hit unreachable code");
        }

        public struct ResultCounter
        {
            public int VerificationErrors;
            public int Verified;
            public int Inconclusives;
            public int TimeOuts;
            public int OutOfMemories;
            public int InputErrors;
            public int InternalErrors; // Should only be used for caught exceptions

            public override string ToString()
            {
                var SB = new StringBuilder();
                SB.AppendFormat("Errors: {0}", VerificationErrors);
                SB.AppendLine();
                SB.AppendFormat("Verified: {0}", Verified);
                SB.AppendLine();
                SB.AppendFormat("Inconclusives: {0}", Inconclusives);
                SB.AppendLine();
                SB.AppendFormat("TimeOuts: {0}", Inconclusives);
                SB.AppendLine();
                SB.AppendFormat("OutOfMemories: {0}", OutOfMemories);
                SB.AppendLine();
                SB.AppendFormat("InputErrors: {0}", InputErrors);
                SB.AppendLine();
                SB.AppendFormat("InternalErrors: {0}", InternalErrors);
                SB.AppendLine();
                return SB.ToString();
            }

            public void Reset()
            {
                VerificationErrors = Verified = Inconclusives = TimeOuts = OutOfMemories = InputErrors = InternalErrors = 0;
            }

            public int TotalErrors
            {
                get
                {
                    return VerificationErrors + Inconclusives + TimeOuts + OutOfMemories + InputErrors + InternalErrors;
                }
            }

            public bool AllVerified()
            {
                // This allows for "Verified" to be zero which is valid for the empty Boogie program
                return TotalErrors == 0;
            }

            // Returns true iff reported errors are verification errors
            public bool AllErrorsAreVerificationErrors()
            {
                return (TotalErrors - VerificationErrors) == 0;
            }

            public bool HasInternalError()
            {
                return InternalErrors > 0;
            }

            public bool HasNonInternalError()
            {
                // We subtract "VerificationErrors" here because its a bug report, not an error within GPUVerifyCruncher/GPUVerifyBoogieDriver
                return (TotalErrors - (InternalErrors + VerificationErrors)) > 0;
            }

            public static ResultCounter GetNewCounterWithInternalError()
            {
                var temp = new ResultCounter();
                temp.InternalErrors = 1;
                return temp;
            }

            public static ResultCounter GetNewCounterWithInputError()
            {
                var temp = new ResultCounter();
                temp.InputErrors = 1;
                return temp;
            }
        }

        public static void ProcessOutcome(Program program, string implName, VC.VCGen.Outcome outcome, List<Counterexample> errors, string timeIndication,
          ref ResultCounter counters)
        {
            switch (outcome)
            {
                default:
                    Contract.Assert(false);  // unexpected outcome
                    throw new cce.UnreachableException();
                case ConditionGeneration.Outcome.ReachedBound:
                    GVUtil.IO.Inform(string.Format("{0}verified", timeIndication));
                    Console.WriteLine(string.Format("Stratified Inlining: Reached recursion bound of {0}", CommandLineOptions.Clo.RecursionBound));
                    counters.Verified++;
                    break;
                case ConditionGeneration.Outcome.Correct:
                    if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed)
                    {
                        GVUtil.IO.Inform(string.Format("{0}credible", timeIndication));
                        counters.Verified++;
                    }
                    else
                    {
                        GVUtil.IO.Inform(string.Format("{0}verified", timeIndication));
                        counters.Verified++;
                    }
                    break;
                case ConditionGeneration.Outcome.TimedOut:
                    counters.TimeOuts++;
                    GVUtil.IO.Inform(string.Format("{0}timed out", timeIndication));
                    break;
                case ConditionGeneration.Outcome.OutOfMemory:
                    counters.OutOfMemories++;
                    GVUtil.IO.Inform(string.Format("{0}out of memory", timeIndication));
                    break;
                case ConditionGeneration.Outcome.Inconclusive:
                    counters.Inconclusives++;
                    GVUtil.IO.Inform(string.Format("{0}inconclusive", timeIndication));
                    break;
                case ConditionGeneration.Outcome.Errors:
                    if (CommandLineOptions.Clo.vcVariety == CommandLineOptions.VCVariety.Doomed)
                    {
                        GVUtil.IO.Inform(string.Format("{0}doomed", timeIndication));
                        counters.VerificationErrors++;
                    }

                    Contract.Assert(errors != null);  // guaranteed by postcondition of VerifyImplementation

                    // BP1xxx: Parsing errors
                    // BP2xxx: Name resolution errors
                    // BP3xxx: Typechecking errors
                    // BP4xxx: Abstract interpretation errors (Is there such a thing?)
                    // BP5xxx: Verification errors

                    errors.Sort(new CounterexampleComparer());

                    foreach (Counterexample error in errors)
                    {
                        new GPUVerifyErrorReporter(program, implName).ReportCounterexample(error);
                        counters.VerificationErrors++;
                    }

                    GVUtil.IO.Inform(string.Format("{0}error{1}", timeIndication, errors.Count == 1 ? "" : "s"));

                    break;
            }
        }
    }
}
