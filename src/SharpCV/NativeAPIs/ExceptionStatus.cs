namespace SharpCV
{
    /// <summary>
    ///     What every <c>CVAPI(ExceptionStatus)</c> entry point of OpenCvSharpExtern returns: whether OpenCV raised an
    ///     error inside the call. Only the Linux / macOS builds ever return <see cref="Occurred"/> — they catch the C++
    ///     exception at the C boundary. The Windows build compiles that catch away, so there the error arrives as a
    ///     <see cref="System.Runtime.InteropServices.SEHException"/> instead; <see cref="NativeMethods"/> turns both into
    ///     an <see cref="OpenCVException"/>.
    /// </summary>
    /// <remarks>Never ignore it: a call that failed leaves its output Mat empty, which is how SharpCV used to return
    ///     silent empty results before the status was checked.</remarks>
    internal enum ExceptionStatus
    {
        /// <summary>The call completed; its out parameters hold valid results.</summary>
        NotOccurred = 0,

        /// <summary>OpenCV raised an error; out parameters may be unset and output arrays empty.</summary>
        Occurred = 1,
    }
}
