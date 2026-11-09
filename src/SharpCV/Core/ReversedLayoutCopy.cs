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
    ///     NumSharp's copy takes 8.7 ms and NumPy's 7.9-16 ms on the same two pinned CPUs (both gather element by
    ///     element, three bytes per inner run), opencv-python's whole <c>cvtColor(frame[..., ::-1], COLOR_RGB2GRAY)</c>
    ///     10.9 ms, while OpenCV swaps the channels of the same frame in 1.8 ms and mirrors it in 3.3 ms, both including
    ///     a fresh allocation.
    ///     </para>
    ///     <para>
    ///     <b>No copy at all.</b> cvtColor absorbs a channel reversal: <see cref="TryChannelReversedSource"/> hands it the
    ///     memory under a 3-channel uint8 / uint16 <c>[..., ::-1]</c> view, and it converts with the code's BGR / RGB
    ///     twin. Every other operation needs the copy: a cv::Mat cannot hold a negative stride, and only cvtColor has a
    ///     twin that reads the reversed order itself.
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
            if (!TryUnreverse(nd, depth, multichannel, strides, out Unreversed layout))
                return null;
            long rows = layout.Rows, cols = layout.Cols, channels = layout.Channels, pixel = layout.Pixel;
            bool rowsReversed = layout.RowsReversed, colsReversed = layout.ColsReversed, channelsReversed = layout.ChannelsReversed;
            // Small arrays stay on NumSharp's path (the route's fixed cost would dominate).
            if (rows * cols * pixel < MinBytes)
                return null;
            // One permutation per copy (flips, or the channel order); none means the copy is needed for another reason
            // (a transpose, a gap, a broadcast), which only NumSharp's gather handles.
            if (channelsReversed ? rowsReversed || colsReversed : !(rowsReversed || colsReversed))
                return null;

            var dims = new[] { (int)rows, (int)cols };
            // flip takes every depth; the channel routes run on a depth their kernels have for the element size (they
            // move bits, so reading 8S as 8U or float16 as 16U changes nothing in the copy).
            bool swap = false;
            MatType type = channelsReversed
                ? MoveDepth(depth, (int)channels, out swap).MakeType((int)channels)
                : depth.MakeType((int)channels);
            var sourceSteps = new[] { (nuint)layout.RowStep, (nuint)pixel };
            var copySteps = new[] { (nuint)(cols * pixel), (nuint)pixel };
            byte* first = layout.First;

            var copy = np.empty(new Shape(nd.Shape.dimensions), nd.dtype);
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

        /// <summary>The channel-reversed views the current thread handed to OpenCV WITHOUT a copy
        ///     (<see cref="TryChannelReversedSource"/>); per thread for the same reason as <see cref="Engaged"/>.</summary>
        [ThreadStatic] private static long _sourcesEngaged;

        /// <summary>How many channel-reversed views the calling thread passed to OpenCV without copying them.</summary>
        internal static long SourcesEngaged => _sourcesEngaged;

        /// <summary>
        ///     A header over the memory UNDER a channel-reversed 3-channel image view (<c>img[..., ::-1]</c>), in the
        ///     memory's own channel order, so that a caller whose operation can absorb the reversal (cvtColor with the
        ///     code's BGR / RGB twin) reads the view without any copy. Null when the view is not exactly that.
        /// </summary>
        /// <param name="nd">The argument.</param>
        /// <returns>The header (the caller disposes it, and keeps <paramref name="nd"/> reachable while OpenCV reads
        ///     through it), or null: not 3-D with 3 channels, not uint8 / uint16, empty, a spatial axis reversed too, or a
        ///     layout cv::Mat cannot view even un-reversed.</returns>
        /// <exception cref="DllNotFoundException">The native library cannot be loaded.</exception>
        /// <exception cref="NotSupportedException">The loaded library is not the verified OpenCV version.</exception>
        /// <remarks>
        ///     Integer depths only. cvtColor's 8U and 16U kernels sum fixed-point products as integers, so a BGR code and
        ///     its RGB twin agree bit for bit (tested on every code with a twin); its float32 kernels add the weighted
        ///     channels in MEMORY order, so the twin rounds differently in the last bit (measured: 8 of 48 BGR2GRAY values
        ///     differed). A float32 view keeps the copy path (OpenCV's channel swap), and so do the depths cvtColor refuses
        ///     (its own error then names them, as cv2's does).
        /// </remarks>
        internal static unsafe Mat TryChannelReversedSource(NDArray nd)
        {
            if (nd.ndim != 3 || !MatDtypes.TryToDepth(nd.typecode, out MatDepth depth)
                || (depth != MatDepth.CV_8U && depth != MatDepth.CV_16U))
                return null;

            Span<long> strides = stackalloc long[3];
            NDArrayToMat.ByteStrides(nd, strides);
            if (!TryUnreverse(nd, depth, multichannel: true, strides, out Unreversed layout)
                || layout.Channels != 3 || !layout.ChannelsReversed || layout.RowsReversed || layout.ColsReversed
                || layout.Rows == 0 || layout.Cols == 0)
                return null;

            try
            {
                var header = Header(new[] { (int)layout.Rows, (int)layout.Cols }, depth.MakeType(3), (IntPtr)layout.First,
                    new[] { (nuint)layout.RowStep, (nuint)layout.Pixel });
                _sourcesEngaged++;
                return header;
            }
            catch (OpenCVException)
            {
                // OpenCV refused a header the gate admitted (it never has): the caller takes the copy path.
                return null;
            }
        }

        /// <summary>
        ///     A reversed view described from the memory under it: the size of each axis, which axes run backwards, the
        ///     first element of the UN-reversed layout, and that layout's row step and pixel size.
        /// </summary>
        private readonly unsafe struct Unreversed
        {
            /// <summary>Rows, columns and channels (1 for a 2-D image).</summary>
            public readonly long Rows, Cols, Channels;

            /// <summary>The un-reversed row step in bytes (the packed value for a single row, as cv2 normalizes it) and
            ///     the pixel size in bytes.</summary>
            public readonly long RowStep, Pixel;

            /// <summary>Which axes the view runs backwards (an axis of length 1 never does).</summary>
            public readonly bool RowsReversed, ColsReversed, ChannelsReversed;

            /// <summary>The un-reversed layout's first element: where each reversed axis ENDS.</summary>
            public readonly byte* First;

            /// <summary>Stores the analysis.</summary>
            /// <param name="rows">Rows.</param>
            /// <param name="cols">Columns.</param>
            /// <param name="channels">Channels.</param>
            /// <param name="rowStep">The un-reversed row step.</param>
            /// <param name="pixel">The pixel size.</param>
            /// <param name="rowsReversed">Rows run backwards.</param>
            /// <param name="colsReversed">Columns run backwards.</param>
            /// <param name="channelsReversed">Channels run backwards.</param>
            /// <param name="first">The un-reversed first element.</param>
            public Unreversed(long rows, long cols, long channels, long rowStep, long pixel,
                bool rowsReversed, bool colsReversed, bool channelsReversed, byte* first)
            {
                Rows = rows;
                Cols = cols;
                Channels = channels;
                RowStep = rowStep;
                Pixel = pixel;
                RowsReversed = rowsReversed;
                ColsReversed = colsReversed;
                ChannelsReversed = channelsReversed;
                First = first;
            }
        }

        /// <summary>
        ///     Describes <paramref name="nd"/> from the memory under it when that UN-reversed layout is one cv::Mat views
        ///     by cv2's own rules (packed channels, pixels packed along a row, descending strides) and fits a Mat's int
        ///     axes; whether any axis actually runs backwards is left to the caller.
        /// </summary>
        /// <param name="nd">A 2-D image, or a 3-D one whose last axis cv2 folds into channels.</param>
        /// <param name="depth">Its OpenCV depth.</param>
        /// <param name="multichannel">cv2's multichannel decision for <paramref name="nd"/>.</param>
        /// <param name="strides"><paramref name="nd"/>'s byte strides.</param>
        /// <param name="layout">The description, when the method returns true.</param>
        /// <returns>False for any other shape or layout.</returns>
        private static unsafe bool TryUnreverse(NDArray nd, MatDepth depth, bool multichannel, ReadOnlySpan<long> strides, out Unreversed layout)
        {
            layout = default;
            int ndims = nd.ndim;
            if (ndims != 2 && !(ndims == 3 && multichannel))
                return false;

            long[] sizes = nd.Shape.dimensions;
            long rows = sizes[0], cols = sizes[1], channels = ndims == 3 ? sizes[2] : 1;
            long esz1 = depth.ElemSize1();
            // Axes a Mat cannot hold stay on NumSharp's path, which refuses them with cv2's message.
            if (rows > int.MaxValue || cols > int.MaxValue)
                return false;

            // An axis of length 1 is never reversed: its stride is never used (NumPy's relaxed strides).
            bool rowsReversed = rows > 1 && strides[0] < 0;
            bool colsReversed = cols > 1 && strides[1] < 0;
            bool channelsReversed = ndims == 3 && channels > 1 && strides[2] < 0;

            long rowStep = Math.Abs(strides[0]), colStep = Math.Abs(strides[1]);
            long pixel = esz1 * channels;
            if (ndims == 3 && channels > 1 && Math.Abs(strides[2]) != esz1)
                return false;
            if (ndims == 3 ? colStep != pixel : cols > 1 && colStep != esz1)
                return false;
            if (rows > 1 && rowStep < colStep)
                return false;

            // The un-reversed layout starts where each reversed axis ENDS (its stride is negative, so this steps back).
            byte* view = nd.Unsafe.Storage.Address + nd.Shape.Offset * nd.dtypesize;
            byte* first = view
                          + (rowsReversed ? (rows - 1) * strides[0] : 0)
                          + (colsReversed ? (cols - 1) * strides[1] : 0)
                          + (channelsReversed ? (channels - 1) * strides[2] : 0);
            // A single row's step is never read; cv2's normalization gives it the packed value, as here.
            layout = new Unreversed(rows, cols, channels, rows > 1 ? rowStep : cols * pixel, pixel,
                rowsReversed, colsReversed, channelsReversed, first);
            return true;
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
