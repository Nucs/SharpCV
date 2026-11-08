using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>new cv::Mat()</c>: an empty Mat that an OpenCV output parameter can allocate into.</summary>
        /// <param name="returnValue">Receives the new Mat; delete it with <see cref="core_Mat_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_new1(out IntPtr returnValue);

        /// <summary><c>new cv::Mat(*mat, roi)</c>: a header over a rectangle of <paramref name="mat"/>'s pixels that SHARES
        ///     them (the reference count goes up when the data is OpenCV-allocated; external data stays owned by its
        ///     owner).</summary>
        /// <param name="mat">The source Mat.</param>
        /// <param name="rect">The region (x, y, width, height); OpenCV asserts it lies inside the Mat.</param>
        /// <param name="returnValue">Receives the new header.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_new7(IntPtr mat, Rect rect, out IntPtr returnValue);

        /// <summary><c>new cv::Mat(rows, cols, type, data, step)</c>: a 2-D header over EXTERNAL memory that the Mat
        ///     neither owns nor reference-counts.</summary>
        /// <param name="rows">The row count.</param>
        /// <param name="cols">The column count.</param>
        /// <param name="type">The element type (depth and channels).</param>
        /// <param name="data">The first pixel.</param>
        /// <param name="step">The bytes between rows; 0 (<c>Mat::AUTO_STEP</c>) means packed rows.</param>
        /// <param name="returnValue">Receives the new header.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_new8(int rows, int cols, MatType type, IntPtr data, nuint step, out IntPtr returnValue);

        /// <summary><c>new cv::Mat(ndims, sizes, type, data, steps)</c>: an n-D header over EXTERNAL memory with explicit
        ///     byte steps (the NDArray -&gt; Mat view). OpenCV reads <c>ndims - 1</c> steps (the last axis is always
        ///     packed) and turns a 1-D request into an N x 1 Mat.</summary>
        /// <param name="ndims">The number of dimensions.</param>
        /// <param name="sizes">The size of each dimension.</param>
        /// <param name="type">The element type (depth and channels).</param>
        /// <param name="data">The first element.</param>
        /// <param name="steps">The byte step of every dimension but the last; each must be a multiple of the depth's
        ///     size.</param>
        /// <param name="returnValue">Receives the new header.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_new9(int ndims, int[] sizes, MatType type, IntPtr data, nuint[] steps, out IntPtr returnValue);

        /// <summary><c>new cv::Mat(*mat)</c>: a second header over the SAME pixels, which takes a reference on
        ///     OpenCV-allocated data. An NDArray view holds one such header as its lease, so the pixels outlive the
        ///     <see cref="Mat"/> the view came from.</summary>
        /// <param name="mat">The source Mat.</param>
        /// <param name="returnValue">Receives the new header; delete it with <see cref="core_Mat_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_new12(IntPtr mat, out IntPtr returnValue);

        /// <summary><c>new cv::Mat(mat-&gt;clone())</c>: a deep copy with packed rows that shares nothing with the
        ///     source.</summary>
        /// <param name="mat">The source Mat.</param>
        /// <param name="returnValue">Receives the copy.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_clone(IntPtr mat, out IntPtr returnValue);

        /// <summary><c>mat.cols</c> (-1 for a Mat of more than two dimensions).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the column count.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_cols(IntPtr mat, out int output);

        /// <summary><c>mat.rows</c> (-1 for a Mat of more than two dimensions).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the row count.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_rows(IntPtr mat, out int output);

        /// <summary><c>mat.ptr(row, col)</c>: the address of one pixel of a 2-D Mat.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="row">The row.</param>
        /// <param name="col">The column.</param>
        /// <param name="output">Receives the pixel's address.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_ptr2d(IntPtr mat, int row, int col, out IntPtr output);

        /// <summary><c>mat.dims</c>: 2 for every image, more for an n-D Mat (dnn blobs), 0 for an empty Mat.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the dimension count.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_dims(IntPtr mat, out int output);

        /// <summary><c>mat.size[dim]</c>.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="dim">The dimension.</param>
        /// <param name="output">Receives its size.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_sizeAt(IntPtr mat, int dim, out int output);

        /// <summary><c>mat.step[dim]</c>: the byte distance between consecutive elements of a dimension (an ROI keeps
        ///     its parent's row step).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="dim">The dimension.</param>
        /// <param name="output">Receives the step in bytes.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_stepAt(IntPtr mat, int dim, out nuint output);

        /// <summary><c>mat.type()</c>: depth and channel count packed as <c>CV_MAKETYPE</c>.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the type.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_type(IntPtr mat, out MatType output);

        /// <summary><c>mat.data</c>: the first pixel (null for an empty Mat).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the address.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_Mat_data(IntPtr mat, out byte* output);

        /// <summary><c>mat.data</c> typed as int (the same export as the byte overload).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the address.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_Mat_data(IntPtr mat, out int* output);

        /// <summary><c>mat.data</c> typed as float (the same export as the byte overload).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the address.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_Mat_data(IntPtr mat, out float* output);

        /// <summary><c>mat.channels()</c>.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the channel count.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_channels(IntPtr mat, out int output);

        /// <summary><c>mat.depth()</c>: the <see cref="MatDepth"/> code.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the depth.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_depth(IntPtr mat, out int output);

        /// <summary><c>delete mat</c>: frees the header and releases its reference on OpenCV-allocated pixels.</summary>
        /// <param name="mat">The Mat to delete; never use it again.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_delete(IntPtr mat);

        /// <summary><c>mat.size()</c> as (width, height) of a 2-D Mat.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the size.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_size(IntPtr mat, out Size output);

        /// <summary><c>mat.elemSize()</c>: bytes per pixel (all channels). The export writes a <c>size_t</c>; SharpCV
        ///     declared it as int until 0.20, which overwrote four bytes past the caller's variable.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the size in bytes.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_elemSize(IntPtr mat, out nuint output);

        /// <summary><c>mat.elemSize1()</c>: bytes per channel value (the depth's size).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the size in bytes.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_elemSize1(IntPtr mat, out nuint output);

        /// <summary><c>mat.datastart</c>: the first byte of the buffer the header references (an ROI's parent buffer
        ///     start, not its own first pixel).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the address (null for an empty Mat).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_Mat_datastart(IntPtr mat, out byte* output);

        /// <summary><c>mat.datalimit</c>: one past the last byte of the buffer the header references, so
        ///     <c>datalimit - datastart</c> is the whole allocation.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the address (null for an empty Mat).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern unsafe ExceptionStatus core_Mat_datalimit(IntPtr mat, out byte* output);

        /// <summary><c>mat.empty()</c>: true when the Mat has no pixels.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives 1 for an empty Mat, 0 otherwise.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_empty(IntPtr mat, out int output);

        /// <summary><c>mat.isContinuous()</c>: true when the rows are packed with no gap (false for most ROIs).</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives 1 for a continuous Mat, 0 otherwise.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_Mat_isContinuous(IntPtr mat, out int output);

        /// <summary><c>new cv::_InputArray(*mat)</c>: the argument wrapper OpenCV functions read a Mat through. It points
        ///     at the Mat's header, so it is valid only while that header lives.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the wrapper; delete it with <see cref="core_InputArray_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_InputArray_new_byMat(IntPtr mat, out IntPtr output);

        /// <summary><c>delete ia</c>: frees an argument wrapper (it never owned the Mat). Each Mat leaked one per wrapper
        ///     kind before SharpCV 0.20.</summary>
        /// <param name="ia">The wrapper.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_InputArray_delete(IntPtr ia);

        /// <summary><c>new cv::_OutputArray(*mat)</c>: the wrapper OpenCV functions write (and may reallocate) a Mat
        ///     through; OpenCvSharp passes it for <c>InputOutputArray</c> parameters too, which add no state.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="output">Receives the wrapper; delete it with <see cref="core_OutputArray_delete"/>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_OutputArray_new_byMat(IntPtr mat, out IntPtr output);

        /// <summary><c>delete oa</c>: frees an output wrapper (it never owned the Mat).</summary>
        /// <param name="oa">The wrapper.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_OutputArray_delete(IntPtr oa);

        /// <summary><c>cv::rotate(src, dst, rotateCode)</c>: rotation by a multiple of 90 degrees.</summary>
        /// <param name="mat">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="flags">The <see cref="RotateFlags"/> code.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_rotate(IntPtr mat, IntPtr dst, int flags);

        /// <summary><c>cv::flip(src, dst, flipCode)</c>.</summary>
        /// <param name="mat">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="mode">The flip code: 0 vertical, positive horizontal, negative both.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_flip(IntPtr mat, IntPtr dst, int mode);

        /// <summary>
        ///     <c>cv::mixChannels(srcVec, dstVec, fromTo, npairs)</c> (OpenCvSharpExtern 4.13 core.h:
        ///     <c>core_mixChannels(cv::Mat** src, uint32_t nsrcs, cv::Mat** dst, uint32_t ndsts, int* fromTo,
        ///     uint32_t npairs)</c>): copies channel <c>fromTo[2k]</c> of the sources into channel <c>fromTo[2k+1]</c> of
        ///     the destinations. The destinations must already be allocated with the right size and depth.
        /// </summary>
        /// <param name="src">The source headers (<c>cv::Mat*</c>, not <c>_InputArray</c>).</param>
        /// <param name="nsrcs">The source count.</param>
        /// <param name="dst">The destination headers.</param>
        /// <param name="ndsts">The destination count.</param>
        /// <param name="fromTo">The channel pairs, flattened.</param>
        /// <param name="npairs">The pair count (half of <paramref name="fromTo"/>'s length).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_mixChannels(IntPtr[] src, uint nsrcs, IntPtr[] dst, uint ndsts, int[] fromTo, uint npairs);

        /// <summary><c>cv::hconcat</c> over an array of Mats.</summary>
        /// <param name="src">The Mats (cv::Mat pointers, not argument wrappers).</param>
        /// <param name="nsrc">Their count.</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_hconcat1([MarshalAs(UnmanagedType.LPArray)] IntPtr[] src, uint nsrc, IntPtr dst);

        /// <summary><c>cv::vconcat</c> over an array of Mats.</summary>
        /// <param name="src">The Mats (cv::Mat pointers, not argument wrappers).</param>
        /// <param name="nsrc">Their count.</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_vconcat1([MarshalAs(UnmanagedType.LPArray)] IntPtr[] src, uint nsrc, IntPtr dst);

        /// <summary><c>cv::hconcat(src1, src2, dst)</c>.</summary>
        /// <param name="mat1">The left input (an <c>_InputArray</c>).</param>
        /// <param name="mat2">The right input (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_hconcat2(IntPtr mat1, IntPtr mat2, IntPtr dst);

        /// <summary><c>cv::split(src, mv)</c>: one single-channel Mat per channel, copied.</summary>
        /// <param name="src">The source Mat (a cv::Mat pointer).</param>
        /// <param name="mv">A caller-allocated <c>std::vector&lt;cv::Mat&gt;</c> that receives the planes.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_split(IntPtr src, IntPtr mv);

        /// <summary><c>cv::vconcat(src1, src2, dst)</c>.</summary>
        /// <param name="mat1">The top input (an <c>_InputArray</c>).</param>
        /// <param name="mat2">The bottom input (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus core_vconcat2(IntPtr mat1, IntPtr mat2, IntPtr dst);
    }
}
