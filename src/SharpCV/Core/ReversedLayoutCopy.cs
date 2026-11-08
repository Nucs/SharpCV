using System;
using System.Threading;
using NumSharp;

namespace SharpCV
{
    /// <summary>
    ///     The C-ordered copy cv2 makes of an image view with REVERSED axes (<c>img[..., ::-1]</c>, the BGR / RGB idiom;
    ///     <c>img[:, ::-1]</c>, mirrored; <c>img[::-1]</c>, upside down; or both spatial axes), made by OpenCV's own kernels
    ///     over the un-reversed memory instead of NumSharp's strided gather. The result is the array <c>nd.copy('C')</c>
    ///     makes, byte for byte: every route here is a pure permutation of the elements (OpenCV's flip, its 3-channel
    ///     swap and mixChannels move bits, they never compute).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why.</b> A cv::Mat has no negative strides, so cv2 copies such a view before OpenCV sees it
    ///     (<c>PyArray_GETCONTIGUOUS</c>), and that copy dominates the call: for a 1080p uint8 <c>[..., ::-1]</c> view
    ///     NumPy's copy takes 8.8 ms and NumSharp's 10.9 ms (an element-wise gather whose inner run is three bytes),
    ///     opencv-python's whole <c>cvtColor(frame[..., ::-1], COLOR_RGB2GRAY)</c> 10.9 ms, while OpenCV swaps the
    ///     channels of the same frame in 1.8 ms and mirrors it in 3.3 ms, both including a fresh allocation.
    ///     </para>
    ///     <para>
    ///     <b>Gate.</b> A 2-D image or a 3-D image whose last axis cv2 folds into channels; ONE kind of reversal (the
    ///     spatial axes, or the channel axis); the un-reversed layout one that cv::Mat views by cv2's own rules; at least
    ///     <see cref="MinBytes"/>. Anything else returns null and the caller makes NumSharp's copy, and so does any
    ///     surprise from OpenCV (an error, or a destination it reallocated instead of filling), so the general path is
    ///     never worse off.
    ///     </para>
    /// </remarks>
    internal static class ReversedLayoutCopy
    {
        /// <summary>
        ///     Below this many bytes NumSharp's gather wins. Measured (Release, two pinned CPUs): the OpenCV route costs
        ///     about 3.2 us whatever the size (two headers, their argument wrappers, the call, the reallocation check),
        ///     NumSharp's copy of a reversed view about 2.1 us plus 1.85 ns per byte, so they cross near 600 bytes; at
        ///     4,800 bytes (40 x 40 x 3) the route already takes 4.3 us against 9.1 us.
        /// </summary>
        internal const long MinBytes = 1024;

        /// <summary>The copies the current thread made here; tests prove with it that the route engaged, since its result
        ///     is indistinguishable from NumSharp's by design. Per thread, so tests running in parallel do not count each
        ///     other's copies (the copy runs synchronously on the caller's thread).</summary>
        [ThreadStatic] private static long _engaged;

        /// <summary>How many reversed views the calling thread copied through OpenCV.</summary>
        internal static long Engaged => _engaged;

        /// <summary>
        ///     The C-ordered copy of <paramref name="nd"/> made by OpenCV when its layout qualifies (see the class
        ///     remarks), or null when it does not.
        /// </summary>
        /// <param name="nd">The view cv2 would copy (no dtype cast is pending: int64 / uint64 take NumSharp's
        ///     cast).</param>
        /// <param name="depth">Its OpenCV depth.</param>
        /// <param name="multichannel">cv2's multichannel decision for <paramref name="nd"/>: a 3-D array whose last axis
        ///     becomes channels.</param>
        /// <param name="strides"><paramref name="nd"/>'s byte strides.</param>
        /// <returns>A new C-contiguous array equal to <c>nd.copy('C')</c> (the caller owns it), or null.</returns>
        /// <exception cref="DllNotFoundException">The native library cannot be loaded.</exception>
        /// <exception cref="NotSupportedException">The loaded library is not the verified OpenCV version.</exception>
        internal static unsafe NDArray TryCopy(NDArray nd, MatDepth depth, bool multichannel, ReadOnlySpan<long> strides)
        {
            int ndims = nd.ndim;
            if (ndims != 2 && !(ndims == 3 && multichannel))
                return null;

            long[] sizes = nd.Shape.dimensions;
            long rows = sizes[0], cols = sizes[1], channels = ndims == 3 ? sizes[2] : 1;
            long esz1 = depth.ElemSize1();
            // Small arrays and axes a Mat cannot hold stay on NumSharp's path (the latter is refused there).
            if (rows * cols * channels * esz1 < MinBytes || rows > int.MaxValue || cols > int.MaxValue)
                return null;

            // An axis of length 1 is never reversed: its stride is never used (NumPy's relaxed strides).
            bool rowsReversed = rows > 1 && strides[0] < 0;
            bool colsReversed = cols > 1 && strides[1] < 0;
            bool channelsReversed = ndims == 3 && channels > 1 && strides[2] < 0;
            // One permutation per copy (flips, or the channel order); none means the copy is needed for another reason
            // (a transpose, a gap, a broadcast), which only NumSharp's gather handles.
            if (channelsReversed ? rowsReversed || colsReversed : !(rowsReversed || colsReversed))
                return null;

            // cv2's viewability rules, applied to the un-reversed layout: packed channels, pixels packed along the row,
            // and descending strides.
            long rowStep = Math.Abs(strides[0]), colStep = Math.Abs(strides[1]);
            long pixel = esz1 * channels;
            if (ndims == 3 && channels > 1 && Math.Abs(strides[2]) != esz1)
                return null;
            if (ndims == 3 ? colStep != pixel : cols > 1 && colStep != esz1)
                return null;
            if (rows > 1 && rowStep < colStep)
                return null;

            // The un-reversed layout starts where each reversed axis ENDS (its stride is negative, so this steps back).
            byte* view = nd.Unsafe.Storage.Address + nd.Shape.Offset * nd.dtypesize;
            byte* first = view
                          + (rowsReversed ? (rows - 1) * strides[0] : 0)
                          + (colsReversed ? (cols - 1) * strides[1] : 0)
                          + (channelsReversed ? (channels - 1) * strides[2] : 0);

            var dims = new[] { (int)rows, (int)cols };
            // flip takes every depth; the channel routes run on a depth their kernels have for the element size (they
            // move bits, so reading 8S as 8U or float16 as 16U changes nothing in the copy).
            bool swap = false;
            MatType type = channelsReversed
                ? MoveDepth(depth, (int)channels, out swap).MakeType((int)channels)
                : depth.MakeType((int)channels);
            // A single row's step is never read; cv2's normalization gives it the packed value, as here.
            var sourceSteps = new[] { (nuint)(rows > 1 ? rowStep : cols * pixel), (nuint)pixel };
            var copySteps = new[] { (nuint)(cols * pixel), (nuint)pixel };

            var copy = np.empty(new Shape(sizes), nd.dtype);
            byte* target = copy.Unsafe.Storage.Address + copy.Shape.Offset * copy.dtypesize;
            try
            {
                using var source = Header(dims, type, (IntPtr)first, sourceSteps);
                using var destination = Header(dims, type, (IntPtr)target, copySteps);
                if (channelsReversed)
                    ReverseChannels(source, destination, swap, (int)channels);
                else
                    Flip(source, destination, rowsReversed && colsReversed ? -1 : rowsReversed ? 0 : 1);

                // OpenCV fills a destination of the right size and type in place; had it reallocated one instead, the
                // copy would hold nothing, so NumSharp's copy is made after all.
                NativeMethods.Check(cv2_native_api.core_Mat_datastart(destination.Handle, out byte* written));
                if (written != target)
                {
                    copy.Dispose();
                    return null;
                }

                _engaged++;
                return copy;
            }
            catch (OpenCVException)
            {
                // OpenCV refused a route the gate admitted (it never has): NumSharp's copy is always right.
                copy.Dispose();
                return null;
            }
            finally
            {
                // The source header points into nd's memory: nd stays reachable until OpenCV is done with it.
                GC.KeepAlive(nd);
            }
        }

        /// <summary>A 2-D header over memory OpenCV does not own (deleting it never frees the pixels).</summary>
        /// <param name="dims">Rows and columns.</param>
        /// <param name="type">The depth and channel count.</param>
        /// <param name="data">The first pixel.</param>
        /// <param name="steps">The row step and the pixel size, in bytes.</param>
        /// <returns>The header, owned by the caller.</returns>
        /// <exception cref="OpenCVException">OpenCV refused the layout.</exception>
        private static Mat Header(int[] dims, MatType type, IntPtr data, nuint[] steps)
        {
            var call = (Dims: dims, Type: type, Data: data, Steps: steps, Handle: IntPtr.Zero);
            NativeMethods.Call(ref call, static (ref (int[] Dims, MatType Type, IntPtr Data, nuint[] Steps, IntPtr Handle) s) =>
                cv2_native_api.core_Mat_new9(2, s.Dims, s.Type, s.Data, s.Steps, out s.Handle));
            return new Mat(call.Handle);
        }

        /// <summary><c>cv::flip</c> from <paramref name="source"/> into the preallocated <paramref name="destination"/>.</summary>
        /// <param name="source">The un-reversed view.</param>
        /// <param name="destination">The copy, same size and type.</param>
        /// <param name="code">0 for reversed rows, 1 for reversed columns, -1 for both.</param>
        /// <exception cref="OpenCVException">OpenCV refused the call.</exception>
        private static void Flip(Mat source, Mat destination, int code)
        {
            NativeMethods.Call(() => cv2_native_api.core_flip(source.InputArray, destination.OutputArray, code));
        }

        /// <summary>
        ///     The depth the channel-reversal headers are typed with, and whether OpenCV's BGR / RGB swap does the work.
        ///     The swap (3 channels of 8U, 16U or 32F, the fastest kernel for the most common case) and
        ///     <c>mixChannels</c> both only move elements, so the data is read through a depth of the same element size
        ///     that the kernel has: 8S as 8U, 16S and float16 as 16U (OpenCV 4.13's mixChannels asserts on CV_16F),
        ///     int32 as 32S, float64 as 64F.
        /// </summary>
        /// <param name="depth">The data's depth.</param>
        /// <param name="channels">The channel count.</param>
        /// <param name="swap">True when cvtColor's BGR / RGB swap reverses the channels, false for mixChannels.</param>
        /// <returns>The header depth.</returns>
        private static MatDepth MoveDepth(MatDepth depth, int channels, out bool swap)
        {
            int esz1 = depth.ElemSize1();
            // float32 keeps its own depth for the swap; int32 goes to mixChannels' integer kernel rather than through a
            // float-typed swap, so no integer bit pattern ever sits in a float register on its way.
            swap = channels == 3 && (esz1 == 1 || esz1 == 2 || depth == MatDepth.CV_32F);
            return esz1 switch
            {
                1 => MatDepth.CV_8U,
                2 => MatDepth.CV_16U,
                4 => depth == MatDepth.CV_32F ? MatDepth.CV_32F : MatDepth.CV_32S,
                _ => MatDepth.CV_64F,
            };
        }

        /// <summary>
        ///     Reverses the channel order from <paramref name="source"/> into the preallocated
        ///     <paramref name="destination"/>, both typed by <see cref="MoveDepth"/>.
        /// </summary>
        /// <param name="source">The un-reversed view.</param>
        /// <param name="destination">The copy, same size and type.</param>
        /// <param name="swap">True for cvtColor's BGR / RGB swap (3 channels), false for mixChannels.</param>
        /// <param name="channels">The channel count (2 to 512).</param>
        /// <exception cref="OpenCVException">OpenCV refused the call.</exception>
        private static void ReverseChannels(Mat source, Mat destination, bool swap, int channels)
        {
            if (swap)
            {
                NativeMethods.Call(() => cv2_native_api.imgproc_cvtColor(source.InputArray, destination.OutputArray,
                    (int)ColorConversionCodes.COLOR_BGR2RGB, 0));
                return;
            }

            var pairs = new int[2 * channels];
            for (int k = 0; k < channels; k++)
            {
                pairs[2 * k] = k;
                pairs[2 * k + 1] = channels - 1 - k;
            }

            var sources = new[] { source.Handle };
            var destinations = new[] { destination.Handle };
            NativeMethods.Call(() => cv2_native_api.core_mixChannels(sources, 1, destinations, 1, pairs, (uint)channels));
        }
    }
}
