using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>cv::imshow(winname, *mat)</c>.</summary>
        /// <param name="winName">The window title (created on first use).</param>
        /// <param name="mat">The image (a cv::Mat pointer).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus highgui_imshow(string winName, IntPtr mat);

        /// <summary><c>cv::namedWindow(winname, flags)</c>.</summary>
        /// <param name="winName">The window title.</param>
        /// <param name="flags">The <see cref="WindowFlags"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus highgui_namedWindow(string winName, int flags);

        /// <summary><c>cv::resizeWindow(winname, width, height)</c>.</summary>
        /// <param name="winName">The window title.</param>
        /// <param name="width">The new width.</param>
        /// <param name="height">The new height.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus highgui_resizeWindow(string winName, int width, int height);

        /// <summary><c>cv::waitKey(delay)</c>.</summary>
        /// <param name="delay">Milliseconds to wait; 0 waits forever.</param>
        /// <param name="output">Receives the pressed key's code, or -1 when none was pressed.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus highgui_waitKey(int delay, out int output);

        /// <summary><c>cv::destroyAllWindows()</c>.</summary>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus highgui_destroyAllWindows();
    }
}
