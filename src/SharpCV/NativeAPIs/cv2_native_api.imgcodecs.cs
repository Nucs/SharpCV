using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>new cv::Mat(cv::imread(filename, flags))</c>: an empty Mat (not an error) when the file is missing
        ///     or cannot be decoded, as in OpenCV.</summary>
        /// <param name="filename">The image path (ANSI-marshalled, as OpenCvSharp does on Windows).</param>
        /// <param name="flags">The <see cref="IMREAD_COLOR"/> flags.</param>
        /// <param name="output">Receives the new Mat.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgcodecs_imread(string filename, int flags, out IntPtr output);

        /// <summary><c>cv::imwrite(filename, *img, params)</c>.</summary>
        /// <param name="filename">The output path; its extension picks the encoder.</param>
        /// <param name="img">The image (a cv::Mat pointer).</param>
        /// <param name="params">Encoder parameter pairs (id, value), e.g. <c>IMWRITE_JPEG_QUALITY, 90</c>.</param>
        /// <param name="paramsLength">The number of ints in <paramref name="params"/>.</param>
        /// <param name="output">Receives 1 when the file was written.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgcodecs_imwrite(string filename, IntPtr img, [In] int[] @params, int paramsLength, out int output);

        /// <summary><c>new cv::Mat(cv::imdecode(Mat(1, bufLength, CV_8U, buf), flags))</c>: decodes an in-memory image
        ///     without copying the input bytes first.</summary>
        /// <param name="buf">The encoded bytes; they must stay valid for the call.</param>
        /// <param name="bufLength">Their count.</param>
        /// <param name="flags">The <see cref="IMREAD_COLOR"/> flags.</param>
        /// <param name="output">Receives the decoded Mat (empty when the bytes are no image).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern ExceptionStatus imgcodecs_imdecode_vector(byte* buf, int bufLength, IMREAD_COLOR flags, out IntPtr output);

        /// <summary><c>new cv::Mat(cv::imdecode(*buf, flags))</c> over any array OpenCV accepts as the encoded bytes (a
        ///     continuous 8-bit single-channel Mat; OpenCV asserts that itself).</summary>
        /// <param name="buf">The encoded bytes (an <c>_InputArray</c>).</param>
        /// <param name="flags">The <see cref="IMREAD_COLOR"/> flags.</param>
        /// <param name="output">Receives the decoded Mat (empty when the bytes are no image).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgcodecs_imdecode_InputArray(IntPtr buf, int flags, out IntPtr output);

        /// <summary><c>cv::imencode(ext, *img, *buf, params)</c>: compresses into a caller-allocated
        ///     <c>std::vector&lt;uchar&gt;</c>. The extension is marshalled as ANSI and must be ASCII.</summary>
        /// <param name="ext">The format extension, e.g. <c>".png"</c>.</param>
        /// <param name="img">The image (an <c>_InputArray</c>).</param>
        /// <param name="buf">The <c>std::vector&lt;uchar&gt;</c> that receives the bytes.</param>
        /// <param name="params">Encoder parameter pairs.</param>
        /// <param name="paramsLength">The number of ints in <paramref name="params"/>.</param>
        /// <param name="returnValue">Receives 1 when encoding succeeded.</param>
        /// <returns>The call status.</returns>
        [Pure, DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, BestFitMapping = false, ThrowOnUnmappableChar = true, ExactSpelling = true)]
        internal static extern ExceptionStatus imgcodecs_imencode_vector(
            [MarshalAs(UnmanagedType.LPStr)] string ext, IntPtr img, IntPtr buf, [In] int[] @params, int paramsLength, out int returnValue);
    }
}
