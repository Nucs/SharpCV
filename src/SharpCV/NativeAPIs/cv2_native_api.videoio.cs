using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>new cv::VideoCapture(filename, apiPreference)</c>.</summary>
        /// <param name="filename">The video file, image sequence or URL.</param>
        /// <param name="api">The <see cref="VideoCaptureAPIs"/> backend.</param>
        /// <param name="output">Receives the capture; delete it with <see cref="videoio_VideoCapture_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_new2(string filename, int api, out IntPtr output);

        /// <summary><c>new cv::VideoCapture(device, apiPreference)</c>.</summary>
        /// <param name="device">The camera index.</param>
        /// <param name="api">The <see cref="VideoCaptureAPIs"/> backend.</param>
        /// <param name="output">Receives the capture.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_new3(int device, int api, out IntPtr output);

        /// <summary><c>obj-&gt;read(*image)</c> into a cv::Mat, which OpenCV reallocates when the frame does not fit.</summary>
        /// <param name="vid">The capture.</param>
        /// <param name="frame">The destination (a cv::Mat pointer).</param>
        /// <param name="output">Receives 1 when a frame was read, 0 at the end of the stream.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_read_Mat(IntPtr vid, IntPtr frame, out int output);

        /// <summary><c>obj-&gt;read(*image)</c> through an <c>_OutputArray</c>: OpenCV decodes into the existing pixels
        ///     when the frame fits them (an NDArray frame buffer included) and reallocates otherwise.</summary>
        /// <param name="vid">The capture.</param>
        /// <param name="image">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="output">Receives 1 when a frame was read, 0 at the end of the stream.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_read_OutputArray(IntPtr vid, IntPtr image, out int output);

        /// <summary><c>obj-&gt;isOpened()</c>.</summary>
        /// <param name="vid">The capture.</param>
        /// <param name="output">Receives 1 when the source opened.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_isOpened(IntPtr vid, out int output);

        /// <summary><c>obj-&gt;release()</c>: closes the source; the object stays valid (and closed).</summary>
        /// <param name="vid">The capture.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_release(IntPtr vid);

        /// <summary><c>delete obj</c>. Before SharpCV 0.15 the capture was freed with <c>Marshal.FreeHGlobal</c>
        ///     instead, which corrupts the C++ heap.</summary>
        /// <param name="vid">The capture; never use it again.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_delete(IntPtr vid);

        /// <summary><c>obj-&gt;get(propId)</c>.</summary>
        /// <param name="vid">The capture.</param>
        /// <param name="propId">The <see cref="VideoCaptureProperties"/> code.</param>
        /// <param name="output">Receives the value (0 when the backend does not support the property).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus videoio_VideoCapture_get(IntPtr vid, int propId, out double output);
    }
}
