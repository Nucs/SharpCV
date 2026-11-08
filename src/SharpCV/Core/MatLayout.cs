using System;
using NumSharp;

namespace SharpCV
{
    /// <summary>
    ///     A Mat's memory layout read through OpenCvSharpExtern, already in NumPy terms: the shape opencv-python gives the
    ///     array (the Mat's dimensions, then the channels when there is more than one), the byte strides (the Mat's own
    ///     steps, so an ROI keeps its parent's row step), the dtype of the depth, and the byte range the elements span.
    /// </summary>
    /// <remarks>
    ///     A snapshot: an OpenCV call that writes the Mat through its <see cref="Mat.OutputArray"/> may reallocate it, so
    ///     read the layout again after any such call instead of keeping one.
    /// </remarks>
    internal readonly unsafe struct MatLayout
    {
        /// <summary>The first element (<c>mat.data</c>); null for an empty Mat.</summary>
        public readonly byte* Data;

        /// <summary>The array shape: <c>size[0..dims)</c>, then the channel count when it exceeds 1.</summary>
        public readonly long[] Shape;

        /// <summary>The byte strides: <c>step[0..dims)</c>, then the depth's size for the channel axis.</summary>
        public readonly long[] ByteStrides;

        /// <summary>The bytes from <see cref="Data"/> to the end of the last element (0 when an axis is empty): the span
        ///     a NumSharp buffer over the Mat must cover.</summary>
        public readonly long ByteExtent;

        /// <summary>The number of channel values (the product of <see cref="Shape"/>).</summary>
        public readonly long ElementCount;

        /// <summary>The Mat's type (depth and channels).</summary>
        public readonly MatType Type;

        /// <summary>The NumSharp dtype of <see cref="Type"/>'s depth.</summary>
        public readonly DType DType;

        /// <summary>Builds a layout; only <see cref="Read"/> calls it.</summary>
        /// <param name="data">The first element.</param>
        /// <param name="shape">The array shape.</param>
        /// <param name="byteStrides">The byte strides.</param>
        /// <param name="byteExtent">The spanned bytes.</param>
        /// <param name="elementCount">The element count.</param>
        /// <param name="type">The Mat type.</param>
        /// <param name="dtype">The dtype.</param>
        private MatLayout(byte* data, long[] shape, long[] byteStrides, long byteExtent, long elementCount, MatType type, DType dtype)
        {
            Data = data;
            Shape = shape;
            ByteStrides = byteStrides;
            ByteExtent = byteExtent;
            ElementCount = elementCount;
            Type = type;
            DType = dtype;
        }

        /// <summary>True when the Mat has no pixels (<c>mat.data == NULL</c>), the case opencv-python returns
        ///     <c>None</c> for.</summary>
        public bool IsEmpty => Data == null;

        /// <summary>
        ///     Reads the layout of a live cv::Mat. The getters it calls are plain field reads that cannot raise, so they
        ///     skip <see cref="NativeMethods.Call"/>'s closure and only check the status.
        /// </summary>
        /// <param name="mat">The cv::Mat pointer.</param>
        /// <returns>The layout; <see cref="IsEmpty"/> for a Mat without data.</returns>
        /// <exception cref="OpenCVException">A getter reported an error (a destroyed Mat).</exception>
        /// <exception cref="NotSupportedException">The Mat's depth has no NumSharp dtype (never for OpenCV 4.13's eight
        ///     depths).</exception>
        public static MatLayout Read(IntPtr mat)
        {
            NativeMethods.Check(cv2_native_api.core_Mat_dims(mat, out int dims));
            NativeMethods.Check(cv2_native_api.core_Mat_type(mat, out MatType type));
            NativeMethods.Check(cv2_native_api.core_Mat_data(mat, out byte* data));
            if (data == null || dims <= 0)
                return new MatLayout(null, new long[] { 0, 0 }, new long[] { 0, 0 }, 0, 0, type, MatDtypes.ToDType(type.Depth()));

            int channels = type.Channels();
            MatDepth depth = type.Depth();
            long elemSize1 = depth.ElemSize1();
            int ndim = dims + (channels > 1 ? 1 : 0);
            var shape = new long[ndim];
            var strides = new long[ndim];
            long count = 1;
            // The last element sits at Σ (size_i - 1) * step_i; the extent adds one whole pixel (all channels) to it.
            long extent = elemSize1 * channels;
            for (int i = 0; i < dims; i++)
            {
                NativeMethods.Check(cv2_native_api.core_Mat_sizeAt(mat, i, out int size));
                NativeMethods.Check(cv2_native_api.core_Mat_stepAt(mat, i, out nuint step));
                shape[i] = size;
                strides[i] = (long)step;
                count *= size;
                if (size > 0)
                    extent += (size - 1) * (long)step;
            }

            if (channels > 1)
            {
                // The channel axis: values of one pixel are packed, one depth size apart.
                shape[dims] = channels;
                strides[dims] = elemSize1;
                count *= channels;
            }

            if (count == 0)
                extent = 0;
            return new MatLayout(data, shape, strides, extent, count, type, MatDtypes.ToDType(depth));
        }
    }

    /// <summary>
    ///     The one-to-one map between OpenCV depths and NumSharp dtypes, plus the dtypes opencv-python casts (int64 and
    ///     uint64 become int32) or refuses (every other one).
    /// </summary>
    internal static class MatDtypes
    {
        /// <summary>The NumSharp dtype a Mat of <paramref name="depth"/> is viewed as.</summary>
        /// <param name="depth">The OpenCV depth.</param>
        /// <returns>uint8, int8, uint16, int16, int32, float32, float64 or float16.</returns>
        /// <exception cref="NotSupportedException"><paramref name="depth"/> is not one of the eight depths.</exception>
        public static DType ToDType(MatDepth depth) => depth switch
        {
            MatDepth.CV_8U => np.uint8,
            MatDepth.CV_8S => np.int8,
            MatDepth.CV_16U => np.uint16,
            MatDepth.CV_16S => np.int16,
            MatDepth.CV_32S => np.int32,
            MatDepth.CV_32F => np.float32,
            MatDepth.CV_64F => np.float64,
            MatDepth.CV_16F => np.float16,
            _ => throw new NotSupportedException($"OpenCV depth {(int)depth} has no NumSharp dtype."),
        };

        /// <summary>
        ///     The OpenCV depth an NDArray of <paramref name="typeCode"/> can be viewed as without a cast (opencv-python's
        ///     <c>NPY_UBYTE ... NPY_DOUBLE, NPY_HALF</c> table).
        /// </summary>
        /// <param name="typeCode">The NDArray's storage type.</param>
        /// <param name="depth">Receives the depth when there is one.</param>
        /// <returns>True for the eight dtypes a Mat can hold.</returns>
        public static bool TryToDepth(NPTypeCode typeCode, out MatDepth depth)
        {
            switch (typeCode)
            {
                case NPTypeCode.Byte: depth = MatDepth.CV_8U; return true;
                case NPTypeCode.SByte: depth = MatDepth.CV_8S; return true;
                case NPTypeCode.UInt16: depth = MatDepth.CV_16U; return true;
                case NPTypeCode.Int16: depth = MatDepth.CV_16S; return true;
                case NPTypeCode.Int32: depth = MatDepth.CV_32S; return true;
                case NPTypeCode.Single: depth = MatDepth.CV_32F; return true;
                case NPTypeCode.Double: depth = MatDepth.CV_64F; return true;
                case NPTypeCode.Half: depth = MatDepth.CV_16F; return true;
                default: depth = default; return false;
            }
        }
    }
}
