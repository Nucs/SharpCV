using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>cv::fastNlMeansDenoising(src, dst, h, templateWindowSize, searchWindowSize)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="h">The filter strength.</param>
        /// <param name="templateWindowSize">The template patch size (odd).</param>
        /// <param name="searchWindowSize">The search window size (odd).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus photo_fastNlMeansDenoising(IntPtr src, IntPtr dst, float h,
            int templateWindowSize, int searchWindowSize);

        /// <summary><c>cv::fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>, 3-channel).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="h">The luminance filter strength.</param>
        /// <param name="hColor">The colour filter strength.</param>
        /// <param name="templateWindowSize">The template patch size (odd).</param>
        /// <param name="searchWindowSize">The search window size (odd).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus photo_fastNlMeansDenoisingColored(IntPtr src, IntPtr dst,
            float h, float hColor, int templateWindowSize, int searchWindowSize);
    }
}
