using System;
using NumSharp;

namespace SharpCV
{
    /// <summary>
    ///     The NDArray -&gt; cv::Mat conversion, ported statement for statement from opencv-python 4.13's
    ///     <c>pyopencv_to(PyObject*, Mat&amp;, const ArgInfo&amp;)</c> (modules/python/src2/cv2_convert.cpp) so SharpCV
    ///     accepts, copies, casts and refuses exactly the arrays <c>cv2</c> does. It plans the Mat; the
    ///     <see cref="Mat"/> constructor builds it with <c>core_Mat_new9</c> over the planned memory.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The rules, in cv2's order: an output argument must be writeable; uint8, int8, uint16, int16, int32, float16,
    ///     float32 and float64 map to a depth, int64 and uint64 are CAST to int32 (wrapping, as numpy's unsafe cast does),
    ///     every other dtype is refused; a copy is needed when the last axis is not packed or a stride grows towards the
    ///     last axis (transposed, reversed, broadcast); a 3-D array whose last axis has at most 512 entries is a 2-D Mat
    ///     of that many channels (unless the argument is an n-D blob), which also needs its columns packed.
    ///     </para>
    ///     <para>
    ///     <see cref="MatCodecMode"/> adds NumSharp's interop rule on top: <see cref="MatCodecMode.View"/> turns every
    ///     copy into a refusal (cv2's own behaviour for output arguments), <see cref="MatCodecMode.Copy"/> copies even
    ///     when a view would do.
    ///     </para>
    /// </remarks>
    internal sealed class NDArrayToMat : IDisposable
    {
        /// <summary><c>CV_MAX_DIM</c>: the most dimensions a Mat has.</summary>
        private const int CV_MAX_DIM = 32;

        /// <summary>
        ///     The array the Mat views: a counted NumSharp view of the caller's array (so disposing the caller's
        ///     reference does not free the pixels under the Mat), the caller's array itself when <see cref="Borrowed"/>,
        ///     or the copy made for it. The plan owns a view or a copy until <see cref="TakeSource"/> hands it to the
        ///     <see cref="Mat"/>; disposing the plan before that releases it.
        /// </summary>
        public NDArray Source { get; private set; }

        /// <summary>
        ///     True when <see cref="Source"/> IS the caller's array, which the plan and the Mat reference but never
        ///     dispose: a Mat made for one call (an <see cref="InputArray"/> argument, a drawing target) lives no longer
        ///     than the caller's own reference, so the counted view, a NumSharp alias that costs about as much as the
        ///     rest of a small call's conversion, buys nothing there.
        /// </summary>
        public bool Borrowed { get; private set; }

        /// <summary>Hands <see cref="Source"/> to the Mat that now views it; the plan no longer releases it.</summary>
        /// <returns>The array the Mat must keep (and dispose unless <see cref="Borrowed"/>).</returns>
        public NDArray TakeSource()
        {
            var source = Source;
            Source = null;
            return source;
        }

        /// <summary>Releases <see cref="Source"/> when no Mat took it (the native header could not be built); a borrowed
        ///     array is the caller's and stays untouched.</summary>
        public void Dispose()
        {
            if (!Borrowed)
                Source?.Dispose();
            Source = null;
        }

        /// <summary>True when <see cref="Source"/> is a copy, so OpenCV's writes never reach the caller's array.</summary>
        public bool Copied { get; private set; }

        /// <summary>The Mat's dimension count after the channel axis was folded into the type.</summary>
        public int Dims { get; private set; }

        /// <summary>The Mat's size per dimension.</summary>
        public int[] Sizes { get; private set; }

        /// <summary>The Mat's byte step per dimension (OpenCV ignores the last one and uses the pixel size).</summary>
        public nuint[] Steps { get; private set; }

        /// <summary>The Mat's type: the depth and the folded channel count.</summary>
        public MatType Type { get; private set; }

        /// <summary>The first element of <see cref="Source"/> (its logical first element, the view's offset
        ///     included).</summary>
        public IntPtr Data { get; private set; }

        /// <summary>
        ///     Plans the Mat for <paramref name="nd"/>.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="mode">Whether the Mat may, must or must not share <paramref name="nd"/>'s memory.</param>
        /// <param name="name">The argument's name in refusal messages (cv2 prints the parameter name: <c>src</c>,
        ///     <c>img</c>).</param>
        /// <param name="output">True when OpenCV writes INTO the array (drawing, in-place reads): the array must be
        ///     writeable and viewable, whatever <paramref name="mode"/> says, because a copy would swallow the
        ///     writes.</param>
        /// <param name="ndMat">True for an n-D blob argument (cv2's <c>CV_ND</c>, e.g. <c>Net.setInput</c>): a 3-D
        ///     array stays three dimensions instead of becoming a multi-channel image.</param>
        /// <param name="borrow">True when the Mat lives for one call only: a viewable array is then referenced as it is
        ///     (<see cref="Borrowed"/>) instead of through a counted view. Never for a Mat the caller keeps: it must
        ///     survive the caller disposing its own reference.</param>
        /// <returns>The plan; the caller owns <see cref="Source"/> unless it is <see cref="Borrowed"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">
        ///     <see cref="ErrorCode.StsBadArg"/>, worded as cv2 words it, when: an output array is read-only; the dtype has
        ///     no depth (bool, uint32, char, decimal, complex, text); the array has 32 or more dimensions or an axis longer
        ///     than <see cref="int.MaxValue"/>; or a copy is needed for an output array or under
        ///     <see cref="MatCodecMode.View"/>.
        /// </exception>
        public static NDArrayToMat Plan(NDArray nd, MatCodecMode mode, string name, bool output, bool ndMat, bool borrow = false)
        {
            if (nd is null)
                throw new ArgumentNullException(name);

            if (output && !nd.Shape.IsWriteable)
                throw OpenCVException.BadArgument($"{name} marked as output argument, but provided NumPy array marked as readonly");

            bool needcopy = false, needcast = false;
            if (!MatDtypes.TryToDepth(nd.typecode, out MatDepth depth))
            {
                if (nd.typecode == NPTypeCode.Int64 || nd.typecode == NPTypeCode.UInt64)
                {
                    // cv2 casts NPY_INT64 / NPY_UINT64 / NPY_LONG to NPY_INT: a Mat has no 64-bit integer depth.
                    needcopy = needcast = true;
                    depth = MatDepth.CV_32S;
                }
                else
                {
                    throw OpenCVException.BadArgument($"{name} data type = {nd.dtype.name} is not supported");
                }
            }

            int ndims = nd.ndim;
            if (ndims >= CV_MAX_DIM)
                throw OpenCVException.BadArgument($"{name} dimensionality (={ndims}) is too high");

            long elemsize = depth.ElemSize1();
            // Shape's own dimension array (shared with the array: only read here) and the byte strides in a stack
            // buffer: nd.shape and nd.strides each allocate a fresh array per call.
            long[] sizes = nd.Shape.dimensions;
            Span<long> strides = stackalloc long[Math.Max(ndims, 1)];
            ByteStrides(nd, strides);

            bool ismultichannel = ndims == 3 && sizes[2] <= MatTypeExtensions.CV_CN_MAX && !ndMat;

            for (int i = ndims - 1; i >= 0 && !needcopy; i--)
            {
                // cv2's comment: these checks handle (a) n-D arrays as well as the 1-D and 2-D cases, (b) transposed
                // arrays, whose strides do not descend, and (c) flipped arrays with negative strides; the size > 1
                // condition skips axes whose stride is meaningless (NumPy's relaxed strides).
                if ((i == ndims - 1 && sizes[i] > 1 && strides[i] != elemsize) ||
                    (i < ndims - 1 && sizes[i] > 1 && strides[i] < strides[i + 1]))
                    needcopy = true;
            }

            int channels = 1;
            if (ismultichannel)
            {
                channels = (int)sizes[ndims - 1];
                ndims--;
                // The pixels' channel values must be packed AND the pixels must be packed along the row: cv::Mat's
                // last dimension is always contiguous.
                if (ndims >= 1 && strides[ndims - 1] != elemsize * sizes[ndims])
                    needcopy = true;
                elemsize *= channels;
            }

            var plan = new NDArrayToMat { Type = depth.MakeType(channels) };
            if (mode == MatCodecMode.Copy && !output)
                needcopy = true;

            NDArray source;
            if (needcopy)
            {
                if (output)
                    throw OpenCVException.BadArgument($"Layout of the output array {name} is incompatible with cv::Mat");
                if (mode == MatCodecMode.View)
                {
                    throw OpenCVException.BadArgument(needcast
                        ? $"{name} data type = {nd.dtype.name} needs a cast to int32, which MatCodecMode.View refuses"
                        : $"Layout of the array {name} is incompatible with cv::Mat, and MatCodecMode.View refuses the copy");
                }

                // PyArray_Cast / PyArray_GETCONTIGUOUS: a C-ordered copy (numpy's PyArray_Cast passes is_f_order = 0).
                source = needcast ? nd.astype(np.int32, true, 'C') : nd.copy('C');
                plan.Copied = true;
                // The copy has the same shape and its own (C-order, possibly int32) strides.
                ByteStrides(source, strides);
            }
            else
            {
                // A call-scoped Mat borrows the caller's array; any other Mat takes a counted view, so the caller may
                // dispose its own reference while the Mat still reads the pixels.
                source = borrow ? nd : nd.view();
                plan.Borrowed = borrow;
            }

            try
            {
                var size = new int[Math.Max(ndims, 1)];
                var step = new nuint[Math.Max(ndims, 1)];
                // Normalize strides of size-1 axes (NPY_RELAXED_STRIDES): give them the step their neighbours imply.
                long defaultStep = elemsize;
                for (int i = ndims - 1; i >= 0; --i)
                {
                    if (sizes[i] > int.MaxValue)
                        throw OpenCVException.BadArgument($"{name} axis {i} has {sizes[i]} elements; a Mat axis holds at most {int.MaxValue}");
                    size[i] = (int)sizes[i];
                    if (size[i] > 1)
                    {
                        step[i] = (nuint)strides[i];
                        defaultStep = strides[i] * size[i];
                    }
                    else
                    {
                        step[i] = (nuint)defaultStep;
                        defaultStep *= size[i];
                    }
                }

                // cv2's degenerate case: a 0-d array is a one-element 1-D Mat (which OpenCV then makes 1 x 1).
                if (ndims == 0)
                {
                    size[0] = 1;
                    step[0] = (nuint)elemsize;
                    ndims = 1;
                }

                plan.Dims = ndims;
                plan.Sizes = size;
                plan.Steps = step;
                unsafe
                {
                    // np.MemoryView.Pointer's formula (the storage base plus the view's offset) without allocating the
                    // MemoryView that nd.data hands out.
                    plan.Data = (IntPtr)(source.Unsafe.Storage.Address + source.Shape.Offset * source.dtypesize);
                }

                plan.Source = source;
                return plan;
            }
            catch
            {
                // The plan never reached a Mat: release the view or copy made above at once (a borrowed array is the
                // caller's).
                if (!plan.Borrowed)
                    source.Dispose();
                throw;
            }
        }

        /// <summary>Writes <paramref name="nd"/>'s byte strides (<see cref="NDArray.strides"/>'s values, the element
        ///     strides times the item size) into <paramref name="strides"/> without allocating.</summary>
        /// <param name="nd">The array.</param>
        /// <param name="strides">At least <c>nd.ndim</c> slots.</param>
        private static void ByteStrides(NDArray nd, Span<long> strides)
        {
            long[] elementStrides = nd.Shape.strides;
            long itemsize = nd.itemsize;
            for (int i = 0; i < elementStrides.Length; i++)
                strides[i] = elementStrides[i] * itemsize;
        }
    }
}
