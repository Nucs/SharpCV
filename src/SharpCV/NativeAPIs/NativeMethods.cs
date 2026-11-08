using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace SharpCV
{
    /// <summary>
    ///     The policy layer over <see cref="cv2_native_api"/>: every OpenCvSharpExtern call that can fail goes through
    ///     <see cref="Call"/>, which turns OpenCV's two error channels into one <see cref="OpenCVException"/>, and the
    ///     first call checks that the loaded native library is the OpenCV version SharpCV's P/Invoke signatures were
    ///     verified against.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Two error channels.</b> OpenCvSharpExtern's Windows build compiles its <c>try</c>/<c>catch</c> wrappers
    ///     away, so a <c>cv::Exception</c> leaves the C function and .NET surfaces it as an <see cref="SEHException"/>
    ///     ("External component has thrown an exception", probed on .NET 8.0.29 and 10.0.1; the library keeps working
    ///     afterwards). The Linux / macOS builds catch it and return <see cref="ExceptionStatus.Occurred"/>. Neither
    ///     carries the message, so a callback registered with OpenCV's <c>cv::redirectError</c> records the error
    ///     (code, function, text, file, line) the moment OpenCV raises it, and <see cref="Call"/> attaches that record
    ///     to the exception it throws.
    ///     </para>
    ///     <para>
    ///     <b>Why the callback never throws.</b> OpenCvSharp throws its managed exception from inside the callback on
    ///     Windows. That works on the calling thread, but an error raised inside an OpenCV <c>parallel_for_</c> body runs
    ///     the callback on an OpenCV worker thread, where a managed exception has no handler and ends the process.
    ///     Recording instead lets OpenCV rethrow the C++ exception on the calling thread, which is safe; the record of a
    ///     worker thread is found through the process-wide <see cref="_lastError"/>.
    ///     </para>
    ///     <para>
    ///     <b>One process-wide callback.</b> <c>cv::redirectError</c> is global to the native library: a second wrapper
    ///     over the same OpenCvSharpExtern (OpenCvSharp itself) that registers its own callback replaces SharpCV's, and
    ///     SharpCV's messages then degrade to the bare <see cref="SEHException"/> text.
    ///     </para>
    /// </remarks>
    internal static class NativeMethods
    {
        /// <summary>The OpenCV major.minor version whose OpenCvSharpExtern exports (OpenCvSharp commit b814c38c,
        ///     runtime package 4.13.0.20261107) every <see cref="cv2_native_api"/> signature was checked against.</summary>
        internal const int VerifiedMajor = 4, VerifiedMinor = 13;

        /// <summary>One error as OpenCV's callback reported it.</summary>
        /// <param name="Status">The OpenCV error code.</param>
        /// <param name="Func">The raising function.</param>
        /// <param name="Message">The error description.</param>
        /// <param name="File">The source file.</param>
        /// <param name="Line">The source line.</param>
        /// <param name="Sequence">The process-wide order of the report, to tell a report made during the current call
        ///     from an older one.</param>
        private sealed record ErrorRecord(ErrorCode Status, string Func, string Message, string File, int Line, long Sequence);

        /// <summary>The error the CURRENT thread's last OpenCV call reported; cleared before every call so a stale
        ///     record (an error OpenCV caught internally) never names a later, unrelated failure.</summary>
        [ThreadStatic] private static ErrorRecord _threadError;

        /// <summary>The last error any thread reported: the fallback for an error raised on an OpenCV worker thread,
        ///     whose callback ran there and not on the thread that receives the exception.</summary>
        private static ErrorRecord _lastError;

        /// <summary>Increments with every report; a call compares it before and after to accept only fresh
        ///     process-wide records.</summary>
        private static long _sequence;

        /// <summary>The registered callback. A static field roots the delegate for the life of the process: native code
        ///     keeps only its function pointer, and a collected delegate would turn the next OpenCV error into a crash.</summary>
        private static readonly cv2_native_api.CvErrorCallback ErrorHandler = OnError;

        /// <summary>0 until <see cref="Initialize"/> succeeded once.</summary>
        private static int _initialized;

        /// <summary>The library's version text (<c>cv::getVersionString()</c>), read once for messages.</summary>
        private static string _versionString;

        /// <summary>
        ///     The loaded OpenCV's version text for <see cref="OpenCVException"/> messages (<c>4.13.0</c>), or
        ///     <c>"unknown"</c> when the native library cannot be asked (a missing runtime package): formatting an exception
        ///     must never throw a second, unrelated one.
        /// </summary>
        internal static string VersionString
        {
            get
            {
                if (_versionString != null)
                    return _versionString;
                try
                {
                    var buffer = new byte[64];
                    unsafe
                    {
                        fixed (byte* p = buffer)
                            cv2_native_api.core_getVersionString(p, buffer.Length);
                    }

                    int length = Array.IndexOf(buffer, (byte)0);
                    _versionString = System.Text.Encoding.ASCII.GetString(buffer, 0, length < 0 ? buffer.Length : length);
                }
                catch (Exception)
                {
                    // DllNotFoundException, EntryPointNotFoundException, SEHException: the message stays usable.
                    return "unknown";
                }

                return _versionString;
            }
        }

        /// <summary>
        ///     Registers the error callback and checks the native library's version, once per process. Every
        ///     <see cref="Call"/> runs it first, so no OpenCV call happens against an unverified library.
        /// </summary>
        /// <exception cref="DllNotFoundException">OpenCvSharpExtern cannot be loaded (no runtime package such as
        ///     OpenCvSharp4.runtime.win is installed).</exception>
        /// <exception cref="NotSupportedException">The loaded OpenCvSharpExtern is not OpenCV 4.13: its exports may have
        ///     other signatures (4.4's findContours allocated its vectors, 4.13's expects the caller's), and a mismatched
        ///     P/Invoke corrupts memory instead of failing.</exception>
        internal static void Initialize()
        {
            if (Volatile.Read(ref _initialized) == 1)
                return;

            // Registering twice is harmless (the same function pointer), so racing first calls need no lock.
            cv2_native_api.redirectError(ErrorHandler, IntPtr.Zero, out _);
            cv2_native_api.core_getVersionMajor(out int major);
            cv2_native_api.core_getVersionMinor(out int minor);
            if (major != VerifiedMajor || minor != VerifiedMinor)
            {
                throw new NotSupportedException(
                    $"SharpCV's P/Invoke signatures are verified against OpenCvSharpExtern for OpenCV {VerifiedMajor}.{VerifiedMinor} " +
                    $"(OpenCvSharp4.runtime.win 4.13.0.20261107), but the loaded library is OpenCV {major}.{minor}. " +
                    "Reference the 4.13 runtime package: OpenCvSharpExtern changes export signatures between versions, and a mismatch corrupts memory.");
            }

            Volatile.Write(ref _initialized, 1);
        }

        /// <summary>
        ///     Runs one OpenCvSharpExtern call that returns <see cref="ExceptionStatus"/> and raises an
        ///     <see cref="OpenCVException"/> carrying OpenCV's message when it fails, on any platform.
        /// </summary>
        /// <param name="nativeCall">The call. It may assign captured locals through <c>out</c> arguments; they hold valid
        ///     values only when this method returns normally.</param>
        /// <exception cref="OpenCVException">OpenCV raised an error inside the call.</exception>
        /// <exception cref="DllNotFoundException">The native library cannot be loaded.</exception>
        /// <exception cref="NotSupportedException">The loaded library is not the verified OpenCV version.</exception>
        /// <remarks>
        ///     The caller must keep every managed wrapper whose handle the call uses reachable until this method returns
        ///     (<c>GC.KeepAlive(mat)</c> after it): the lambda captures raw handles only, and a <see cref="Mat"/> finalized
        ///     mid-call frees the cv::Mat OpenCV is working on.
        /// </remarks>
        internal static void Call(Func<ExceptionStatus> nativeCall)
        {
            Initialize();
            _threadError = null;
            long before = Interlocked.Read(ref _sequence);
            ExceptionStatus status;
            try
            {
                status = nativeCall();
            }
            catch (SEHException e)
            {
                // Windows: the C++ exception crossed the P/Invoke boundary; the callback has recorded it already.
                throw TakeError(before, e);
            }

            if (status != ExceptionStatus.NotOccurred)
                throw TakeError(before, null);
        }

        /// <summary>
        ///     Checks the status of an export that cannot raise in practice (a header field getter such as
        ///     <c>core_Mat_dims</c>), without <see cref="Call"/>'s closure: the layout of a Mat is read with up to a dozen
        ///     such calls per NDArray view. A failure still becomes an <see cref="OpenCVException"/>.
        /// </summary>
        /// <param name="status">The export's status.</param>
        /// <exception cref="OpenCVException">The status reports an error.</exception>
        internal static void Check(ExceptionStatus status)
        {
            if (status != ExceptionStatus.NotOccurred)
                throw TakeError(long.MaxValue, null);
        }

        /// <summary>
        ///     Builds the exception for a failed call from the freshest record: the calling thread's own, else one any
        ///     thread made after the call began (an OpenCV worker thread), else a generic one naming the transport.
        /// </summary>
        /// <param name="before">The report sequence read before the call started.</param>
        /// <param name="transport">The <see cref="SEHException"/> that carried the failure on Windows, or null.</param>
        /// <returns>The exception to throw.</returns>
        private static OpenCVException TakeError(long before, Exception transport)
        {
            var record = _threadError;
            _threadError = null;
            if (record == null)
            {
                var last = Volatile.Read(ref _lastError);
                if (last != null && last.Sequence > before)
                    record = last;
            }

            if (record == null)
            {
                // A C++ exception that did not go through cv::error (std::bad_alloc inside OpenCV, for one).
                return new OpenCVException(ErrorCode.StsError, "", transport?.Message ?? "OpenCV reported an error without details", "", -1, transport);
            }

            return new OpenCVException(record.Status, record.Func, record.Message, record.File, record.Line, transport);
        }

        /// <summary>
        ///     OpenCV's error callback: records the error for the thread that raised it and process-wide, then returns so
        ///     OpenCV throws its C++ exception as usual.
        /// </summary>
        /// <param name="status">The OpenCV error code.</param>
        /// <param name="funcName">The raising function (a C string, may be null).</param>
        /// <param name="errMsg">The description (a C string, may be null).</param>
        /// <param name="fileName">The source file (a C string, may be null).</param>
        /// <param name="line">The source line.</param>
        /// <param name="userData">Unused (registered as null).</param>
        /// <returns>0, which OpenCV ignores (it always throws after the callback).</returns>
        private static int OnError(int status, IntPtr funcName, IntPtr errMsg, IntPtr fileName, int line, IntPtr userData)
        {
            // An exception escaping a reverse P/Invoke unwinds through native frames: fatal off Windows, and fatal on any
            // OpenCV worker thread. Nothing here may throw, so a failure to decode only loses the details.
            try
            {
                long sequence = Interlocked.Increment(ref _sequence);
                var record = new ErrorRecord((ErrorCode)status,
                    Marshal.PtrToStringUTF8(funcName) ?? "", Marshal.PtrToStringUTF8(errMsg) ?? "",
                    Marshal.PtrToStringUTF8(fileName) ?? "", line, sequence);
                _threadError = record;
                Volatile.Write(ref _lastError, record);
            }
            catch
            {
                // Swallowed on purpose (see above).
            }

            return 0;
        }
    }
}
