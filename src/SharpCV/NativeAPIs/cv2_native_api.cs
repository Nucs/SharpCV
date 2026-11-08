using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     The raw P/Invoke surface of OpenCvSharpExtern (the C layer of OpenCvSharp's native runtime packages). Every
    ///     declaration mirrors the export's C signature in OpenCvSharp commit b814c38c (the 4.13.0.20261107 runtime):
    ///     a <c>CVAPI(ExceptionStatus)</c> export returns <see cref="ExceptionStatus"/>, which callers route through
    ///     <see cref="NativeMethods.Call"/>; the <c>std::vector</c> helpers return their value directly and cannot fail.
    /// </summary>
    /// <remarks>
    ///     A declaration whose parameter types differ from the export's does not fail: it corrupts memory or reads
    ///     garbage (an int <see cref="Point"/> passed where <c>MyCvPoint2D32f</c> is expected became a rotation centre of
    ///     about 5e-43). Check the export's header before changing any signature here.
    /// </remarks>
    internal partial class cv2_native_api
    {
        /// <summary>The native library name: <c>OpenCvSharpExtern.dll</c> / <c>libOpenCvSharpExtern.so</c>, shipped by
        ///     the OpenCvSharp4 runtime packages (Windows: OpenCvSharp4.runtime.win).</summary>
        internal const string OpenCvDllName = "OpenCvSharpExtern";

        /// <summary>
        ///     OpenCV's <c>cv::ErrorCallback</c>: called by <c>cv::error</c> with the error's details just before OpenCV
        ///     throws its C++ exception. The strings are raw C strings so a decoding problem can never throw inside native
        ///     code.
        /// </summary>
        /// <param name="status">The OpenCV error code.</param>
        /// <param name="funcName">The raising function.</param>
        /// <param name="errMsg">The error description.</param>
        /// <param name="fileName">The source file.</param>
        /// <param name="line">The source line.</param>
        /// <param name="userData">The pointer given to <see cref="redirectError"/>.</param>
        /// <returns>Ignored by OpenCV 4.x.</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int CvErrorCallback(int status, IntPtr funcName, IntPtr errMsg, IntPtr fileName, int line, IntPtr userData);

        /// <summary>Replaces OpenCV's process-wide error callback (<c>cv::redirectError</c>). The delegate must stay
        ///     reachable for as long as OpenCV may call it.</summary>
        /// <param name="errCallback">The new callback.</param>
        /// <param name="userdata">The pointer OpenCV passes back to the callback.</param>
        /// <param name="prevUserdata">Receives the previous callback's user data.</param>
        /// <returns>The previous callback's function pointer.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern IntPtr redirectError(CvErrorCallback errCallback, IntPtr userdata, out IntPtr prevUserdata);

        /// <summary><c>cv::getVersionMajor()</c>: the loaded OpenCV's major version.</summary>
        /// <param name="returnValue">Receives the major version (4 for the verified 4.13 runtime).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_getVersionMajor(out int returnValue);

        /// <summary><c>cv::getVersionMinor()</c>: the loaded OpenCV's minor version.</summary>
        /// <param name="returnValue">Receives the minor version (13 for the verified runtime).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_getVersionMinor(out int returnValue);

        /// <summary><c>cv::getVersionString()</c> copied into a caller buffer, truncated to its length.</summary>
        /// <param name="buf">The buffer that receives the NUL-terminated text.</param>
        /// <param name="bufLength">The buffer's length in bytes.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_getVersionString(byte* buf, int bufLength);
    }
}
