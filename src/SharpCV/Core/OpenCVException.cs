using System;
using System.Runtime.InteropServices;

namespace SharpCV
{
    /// <summary>
    ///     An error OpenCV raised (or SharpCV raised on OpenCV's behalf), the C# counterpart of opencv-python's
    ///     <c>cv2.error</c>. <see cref="Exception.Message"/> is OpenCV's own text, byte for byte the
    ///     <c>cv::Exception::what()</c> that Python prints (minus its trailing newline):
    ///     <c>OpenCV(4.13.0) file.cpp:92: error: (-215:Assertion failed) scn == 3 in function 'cvtColor'</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Before SharpCV 0.20 a failing call returned an empty <see cref="Mat"/> silently (the native status was
    ///     ignored); every wrapper now raises this exception instead, so code that relied on getting an empty result for
    ///     bad input must catch it.
    ///     </para>
    ///     <para>
    ///     SharpCV's own refusals that opencv-python raises from its argument parser (an NDArray dtype OpenCV has no depth
    ///     for, an output array whose layout would need a copy, a read-only output array) use the same type with
    ///     <see cref="ErrorCode.StsBadArg"/>, an empty <see cref="FuncName"/> and line -1, as Python's binding layer
    ///     does (<c>OpenCV(4.13.0) :-1: error: (-5:Bad argument) ...</c>).
    ///     </para>
    /// </remarks>
    [Serializable]
    public class OpenCVException : Exception
    {
        /// <summary>The OpenCV error code (<c>cv::Exception::code</c>).</summary>
        public ErrorCode Status { get; }

        /// <summary>The function that raised the error, as the native build spelled it (MSVC builds give the full
        ///     signature); empty for SharpCV's own argument refusals.</summary>
        public string FuncName { get; }

        /// <summary>The bare error description (<c>cv::Exception::err</c>), already prefixed with <c>"> "</c> per line
        ///     when OpenCV formatted a multi-line message.</summary>
        public string ErrMsg { get; }

        /// <summary>The OpenCV source file that raised the error; empty for SharpCV's own argument refusals.</summary>
        public string FileName { get; }

        /// <summary>The line in <see cref="FileName"/>; -1 for SharpCV's own argument refusals.</summary>
        public int Line { get; }

        /// <summary>
        ///     Creates the exception for an error OpenCV reported through its error callback, formatting the message the
        ///     way <c>cv::Exception::formatMessage</c> does so it reads exactly as opencv-python's <c>cv2.error</c>.
        /// </summary>
        /// <param name="status">The OpenCV error code.</param>
        /// <param name="funcName">The raising function; null or empty when OpenCV gave none.</param>
        /// <param name="errMsg">The error description as the callback received it.</param>
        /// <param name="fileName">The source file; null becomes empty.</param>
        /// <param name="line">The source line.</param>
        /// <param name="innerException">The <see cref="SEHException"/> that carried the C++ exception into .NET on
        ///     Windows, or null.</param>
        public OpenCVException(ErrorCode status, string funcName, string errMsg, string fileName, int line, Exception innerException = null)
            : base(FormatMessage(status, funcName ?? "", errMsg ?? "", fileName ?? "", line), innerException)
        {
            Status = status;
            FuncName = funcName ?? "";
            ErrMsg = errMsg ?? "";
            FileName = fileName ?? "";
            Line = line;
        }

        /// <summary>
        ///     Creates one of SharpCV's own argument refusals, worded as opencv-python's argument parser words it
        ///     (<c>OpenCV(4.13.0) :-1: error: (-5:Bad argument) src data type = bool is not supported</c>).
        /// </summary>
        /// <param name="message">The refusal, e.g. <c>"src data type = bool is not supported"</c>.</param>
        /// <returns>The exception, with <see cref="ErrorCode.StsBadArg"/>, no function, no file and line -1.</returns>
        internal static OpenCVException BadArgument(string message)
            => new OpenCVException(ErrorCode.StsBadArg, "", message, "", -1);

        /// <summary>
        ///     Formats an error exactly as OpenCV 4.13's <c>cv::Exception::formatMessage</c> (core/src/system.cpp) does,
        ///     without the trailing newline OpenCV appends to single-line messages.
        /// </summary>
        /// <param name="status">The error code; its <c>cvErrorStr</c> name goes in parentheses.</param>
        /// <param name="func">The raising function, or empty.</param>
        /// <param name="err">The description; OpenCV has already turned a multi-line one into <c>"> "</c> lines.</param>
        /// <param name="file">The source file.</param>
        /// <param name="line">The source line.</param>
        /// <returns>The formatted message.</returns>
        private static string FormatMessage(ErrorCode status, string func, string err, string file, int line)
        {
            // formatMessage rewrites a description that contains '\n' into "> "-prefixed lines BEFORE the error
            // callback sees it, so a multi-line description is recognized by that prefix here.
            bool multiline = err.StartsWith("> ", StringComparison.Ordinal) && err.IndexOf('\n') >= 0;
            string head = $"OpenCV({NativeMethods.VersionString}) {file}:{line}: error: ({(int)status}:{ErrorStr(status)})";
            string message;
            if (func.Length > 0)
                message = multiline ? $"{head} in function '{func}'\n{err}" : $"{head} {err} in function '{func}'";
            else
                message = $"{head} {err}";
            return message.TrimEnd('\n');
        }

        /// <summary>
        ///     OpenCV's <c>cvErrorStr</c>: the short name printed after the code (<c>Assertion failed</c>), including its
        ///     fallback text for codes it has no name for.
        /// </summary>
        /// <param name="status">The error code.</param>
        /// <returns>The name OpenCV prints for it.</returns>
        internal static string ErrorStr(ErrorCode status) => status switch
        {
            ErrorCode.StsOk => "No Error",
            ErrorCode.StsBackTrace => "Backtrace",
            ErrorCode.StsError => "Unspecified error",
            ErrorCode.StsInternal => "Internal error",
            ErrorCode.StsNoMem => "Insufficient memory",
            ErrorCode.StsBadArg => "Bad argument",
            ErrorCode.StsNoConv => "Iterations do not converge",
            ErrorCode.StsAutoTrace => "Autotrace call",
            ErrorCode.StsBadSize => "Incorrect size of input array",
            ErrorCode.StsNullPtr => "Null pointer",
            ErrorCode.StsDivByZero => "Division by zero occurred",
            ErrorCode.BadStep => "Image step is wrong",
            ErrorCode.StsInplaceNotSupported => "Inplace operation is not supported",
            ErrorCode.StsObjectNotFound => "Requested object was not found",
            ErrorCode.BadDepth => "Input image depth is not supported by function",
            ErrorCode.StsUnmatchedFormats => "Formats of input arguments do not match",
            ErrorCode.StsUnmatchedSizes => "Sizes of input arguments do not match",
            ErrorCode.StsOutOfRange => "One of the arguments' values is out of range",
            ErrorCode.StsUnsupportedFormat => "Unsupported format or combination of formats",
            ErrorCode.BadCOI => "Input COI is not supported",
            ErrorCode.BadNumChannels => "Bad number of channels",
            ErrorCode.StsBadFlag => "Bad flag (parameter or structure field)",
            ErrorCode.StsBadPoint => "Bad parameter of type CvPoint",
            ErrorCode.StsBadMask => "Bad type of mask argument",
            ErrorCode.StsParseError => "Parsing error",
            ErrorCode.StsNotImplemented => "The function/feature is not implemented",
            ErrorCode.StsBadMemBlock => "Memory block has been corrupted",
            ErrorCode.StsAssert => "Assertion failed",
            ErrorCode.GpuNotSupported => "No CUDA support",
            ErrorCode.GpuApiCallError => "Gpu API call",
            ErrorCode.OpenGlNotSupported => "No OpenGL support",
            ErrorCode.OpenGlApiCallError => "OpenGL API call",
            _ => $"Unknown {((int)status >= 0 ? "status" : "error")} code {(int)status}",
        };
    }
}
