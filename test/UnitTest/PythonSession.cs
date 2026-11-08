using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace UnitTest
{
    /// <summary>
    ///     The live parity oracle: ONE embedded CPython per test process running opencv-python 4.13 and NumPy, which the
    ///     parity tests call on the very arrays SharpCV processes (NumSharp.Interop.pythonnet crosses them without copies)
    ///     and compare against bit for bit.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Finding Python.</b> <c>SHARPCV_PYTHON</c> names the interpreter (binding: no other is tried when it is set),
    ///     else <c>python</c> / <c>python3</c> on PATH are probed. The interpreter must import numpy and cv2, and cv2 must
    ///     be 4.13: OpenCvSharp4.runtime.win 4.13 bundles OpenCV 4.13, and comparing across OpenCV versions would test
    ///     OpenCV's changes instead of SharpCV.
    ///     </para>
    ///     <para>
    ///     <b>Without Python.</b> Parity tests report Inconclusive, unless <c>SHARPCV_REQUIRE_PYTHON=1</c> or
    ///     <c>CI=true</c> is set: then a missing oracle FAILS, so a CI run cannot go green by skipping every comparison.
    ///     </para>
    ///     <para>
    ///     <b>Engine rules</b> (measured in NumSharp's interop suite): <see cref="PythonEngine.ProgramName"/> is set for
    ///     every interpreter (otherwise a venv is not adopted and Windows reports the .NET host as sys.executable), and
    ///     <c>gc.collect(); gc.freeze()</c> runs before <see cref="PythonEngine.Shutdown"/>, which otherwise spends
    ///     seconds in 87 full collections.
    ///     </para>
    /// </remarks>
    [TestClass]
    public sealed class PythonSession
    {
        /// <summary>True when the engine started with numpy and cv2 4.13.</summary>
        public static bool Available { get; private set; }

        /// <summary>Why the oracle is unavailable; null when it is available.</summary>
        public static string Reason { get; private set; } = "not initialized";

        /// <summary>The embedded interpreter's executable (diagnostics).</summary>
        public static string Interpreter { get; private set; }

        /// <summary>The cv2 version the oracle runs (diagnostics).</summary>
        public static string Cv2Version { get; private set; }

        /// <summary>
        ///     Starts the engine once for the whole test assembly: CPython with numpy cannot initialize again after
        ///     <c>Py_Finalize</c>, so the assembly owns the lifecycle.
        /// </summary>
        /// <param name="_">The MSTest context (unused).</param>
        [AssemblyInitialize]
        public static void Start(TestContext _)
        {
            try
            {
                if (!Discover(out string exe, out string dll, out string version, out string reason))
                {
                    Reason = reason;
                    return;
                }

                Interpreter = exe;
                Cv2Version = version;
                Runtime.PythonDLL = dll;
                PythonEngine.ProgramName = exe;
                PythonEngine.Initialize();
                PythonEngine.BeginAllowThreads();
                using (Py.GIL())
                {
                    // Process-wide by design: this test host owns the process, and every parity test wants NDArray
                    // arguments to cross as numpy views and numpy results to come back as NDArray views.
                    NDArrayPythonInterop.RegisterCodec();
                    using var cv2 = Py.Import("cv2");
                    using var np = Py.Import("numpy");
                }

                Available = true;
                Reason = null;
            }
            catch (Exception e)
            {
                Available = false;
                Reason = $"engine failed to start: {e.GetType().Name}: {e.Message}";
            }
        }

        /// <summary>Shuts the engine down at the end of the run, after freezing the heap so shutdown's collection
        ///     passes have nothing to scan.</summary>
        [AssemblyCleanup]
        public static void Stop()
        {
            if (!Available)
                return;

            try
            {
                using (Py.GIL())
                {
                    using var gc = Py.Import("gc");
                    using var collected = gc.InvokeMethod("collect");
                    using var frozen = gc.InvokeMethod("freeze");
                }

                // pythonnet 3.0.x stashes runtime state with BinaryFormatter, which .NET 8+ removed; NoopFormatter is
                // pythonnet's own opt-out.
                RuntimeData.FormatterType = typeof(NoopFormatter);
                PythonEngine.Shutdown();
            }
            catch (Exception e)
            {
                Trace.WriteLine($"PythonEngine.Shutdown reported: {e.Message}");
            }
        }

        /// <summary>
        ///     Skips (Inconclusive) or fails the calling test when the oracle is unavailable; see the class remarks for
        ///     which.
        /// </summary>
        /// <exception cref="AssertInconclusiveException">No oracle on a developer machine.</exception>
        /// <exception cref="AssertFailedException">No oracle where one is required.</exception>
        public static void EnsureOrInconclusive()
        {
            if (Available)
                return;

            bool required = IsTrue("SHARPCV_REQUIRE_PYTHON") || IsTrue("CI");
            if (required)
                Assert.Fail($"The opencv-python parity oracle is required (SHARPCV_REQUIRE_PYTHON / CI) but unavailable: {Reason}");
            Assert.Inconclusive($"The opencv-python parity oracle is unavailable: {Reason}. Set SHARPCV_PYTHON to a Python with numpy and opencv-python 4.13.");
        }

        /// <summary>Whether an environment variable is set to a true value.</summary>
        /// <param name="name">The variable.</param>
        /// <returns>True for 1 / true / yes / on.</returns>
        private static bool IsTrue(string name)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return value is not null && (value == "1" || value.Equals("true", StringComparison.OrdinalIgnoreCase)
                                         || value.Equals("yes", StringComparison.OrdinalIgnoreCase)
                                         || value.Equals("on", StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>Finds the interpreter to embed: <c>SHARPCV_PYTHON</c> (binding), else <c>python</c>, else
        ///     <c>python3</c>.</summary>
        /// <param name="exe">Receives the interpreter's sys.executable.</param>
        /// <param name="dll">Receives the shared library pythonnet loads.</param>
        /// <param name="version">Receives cv2.__version__.</param>
        /// <param name="reason">Receives why nothing qualified.</param>
        /// <returns>True when an interpreter qualified.</returns>
        private static bool Discover(out string exe, out string dll, out string version, out string reason)
        {
            exe = dll = version = null;
            string named = Environment.GetEnvironmentVariable("SHARPCV_PYTHON");
            var candidates = string.IsNullOrEmpty(named) ? new[] { "python", "python3" } : new[] { named };
            var attempts = new List<string>();
            foreach (string candidate in candidates)
            {
                if (Probe(candidate, out exe, out dll, out version, out string why))
                {
                    reason = null;
                    return true;
                }

                attempts.Add($"{candidate}: {why}");
            }

            reason = string.Join(" | ", attempts) + (string.IsNullOrEmpty(named) ? "" : " (SHARPCV_PYTHON is binding)");
            return false;
        }

        /// <summary>
        ///     Runs a candidate interpreter once to learn its executable, base prefix and version, and to check that numpy
        ///     and cv2 4.13 import.
        /// </summary>
        /// <param name="candidate">A name on PATH or a path.</param>
        /// <param name="exe">Receives sys.executable.</param>
        /// <param name="dll">Receives the library path (pythonXY.dll under sys.base_prefix on Windows).</param>
        /// <param name="version">Receives cv2.__version__.</param>
        /// <param name="why">Receives the rejection reason.</param>
        /// <returns>True when the candidate qualifies.</returns>
        private static bool Probe(string candidate, out string exe, out string dll, out string version, out string why)
        {
            exe = dll = version = why = null;
            const string script =
                "import sys, sysconfig, numpy, cv2;" +
                "print(sys.executable);" +
                "print(sys.base_prefix);" +
                "print(sys.version_info.major);" +
                "print(sys.version_info.minor);" +
                "print(cv2.__version__);" +
                "print(sysconfig.get_config_var('INSTSONAME') or '');" +
                "print(sysconfig.get_config_var('LIBDIR') or '')";
            try
            {
                var psi = new ProcessStartInfo(candidate, $"-c \"{script}\"")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var proc = Process.Start(psi);
                string stdout = proc.StandardOutput.ReadToEnd();
                string stderr = proc.StandardError.ReadToEnd();
                if (!proc.WaitForExit(20_000))
                {
                    try { proc.Kill(); } catch { }
                    why = "probe timed out";
                    return false;
                }

                if (proc.ExitCode != 0)
                {
                    why = $"probe failed: {stderr.Trim().Split('\n')[^1]}";
                    return false;
                }

                string[] lines = stdout.Replace("\r", "").Split('\n');
                exe = lines[0].Trim();
                string basePrefix = lines[1].Trim();
                int major = int.Parse(lines[2].Trim());
                int minor = int.Parse(lines[3].Trim());
                version = lines[4].Trim();
                if (!version.StartsWith("4.13.", StringComparison.Ordinal))
                {
                    why = $"cv2 {version} is not 4.13, the OpenCV that OpenCvSharp4.runtime.win 4.13 bundles";
                    return false;
                }

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    dll = Path.Combine(basePrefix, $"python{major}{minor}.dll");
                }
                else
                {
                    string soname = lines.Length > 5 ? lines[5].Trim() : "";
                    string libdir = lines.Length > 6 ? lines[6].Trim() : "";
                    dll = Path.Combine(libdir, soname);
                }

                if (!File.Exists(dll))
                {
                    why = $"the Python library '{dll}' does not exist";
                    return false;
                }

                return true;
            }
            catch (Exception e)
            {
                why = $"{e.GetType().Name}: {e.Message}";
                return false;
            }
        }
    }
}
