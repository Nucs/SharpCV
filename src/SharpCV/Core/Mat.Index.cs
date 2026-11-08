using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp;

namespace SharpCV
{
    public partial class Mat
    {
        /// <summary>
        ///     A region of interest that SHARES this Mat's pixels (<c>img[y0:y1, x0:x1]</c> in cv2): drawing on it draws on
        ///     this Mat. Its NDArray view keeps the parent's row stride.
        /// </summary>
        /// <param name="y">The row range [start, end).</param>
        /// <param name="x">The column range [start, end).</param>
        /// <returns>A new Mat header over the region; dispose it independently of this one.</returns>
        /// <exception cref="OpenCVException">The region does not lie inside the Mat.</exception>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public unsafe Mat this[(int, int) y, (int, int) x]
        {
            get
            {
                // var cropped = data[new Slice(y.Item1, y.Item2), new Slice(x.Item1, x.Item2)];
                var source = Handle;
                var rect = new Rect(x.Item1, y.Item1, x.Item2 - x.Item1, y.Item2 - y.Item1);
                IntPtr handle = IntPtr.Zero;
                NativeMethods.Call(() => cv2_native_api.core_Mat_new7(source, rect, out handle));
                // cv2_native_api.imgcodecs_imdecode_vector(cropped.GetData<byte>().ToArray(), cropped.size, IMREAD_COLOR.IMREAD_COLOR, out var handle);
                var roi = new Mat(handle);
                roi.InheritSource(this);
                GC.KeepAlive(this);
                return roi;
            }
        }

        /// <summary>
        ///     One pixel as an NDArray VIEW: a 0-d array for a single-channel image (read it with <c>(byte)img[8, 8]</c>),
        ///     a 1-D array of the channel values for a multi-channel one (<c>[32, 19, 11]</c> for a BGR pixel). Setting
        ///     writes the value(s) into the pixel, broadcasting a scalar over the channels.
        /// </summary>
        /// <param name="row">The row (negative counts from the end, as NumSharp indexing does).</param>
        /// <param name="col">The column.</param>
        /// <returns>The pixel view.</returns>
        /// <exception cref="IndexOutOfRangeException">The index lies outside the image.</exception>
        /// <exception cref="InvalidOperationException">The Mat is empty.</exception>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public unsafe NDArray this[int row, int col]
        {
            get
            {
                // A sub-view of the Mat's own view: NumSharp indexing only, and the pixel holds its own reference.
                var cache = Cache ?? throw new InvalidOperationException("The Mat is empty.");
                return cache.View[row, col];
            }

            set
            {
                // BGR format
                var cache = Cache ?? throw new InvalidOperationException("The Mat is empty.");
                cache.View[row, col] = value;
            }
        }

        /// <summary>
        ///     One channel value of an 8-bit image's pixel, read straight from memory through the cached layout (no native
        ///     call, no NDArray).
        /// </summary>
        /// <param name="row">The row, 0 to rows - 1.</param>
        /// <param name="col">The column, 0 to cols - 1.</param>
        /// <param name="channel">The channel, 0 to <see cref="Channels"/> - 1 (0 = blue in a BGR image).</param>
        /// <returns>The value.</returns>
        /// <exception cref="IndexOutOfRangeException">The row, column or channel lies outside the image; OpenCV's
        ///     <c>ptr</c> checked nothing in release builds, so this used to read beyond the pixels.</exception>
        /// <exception cref="InvalidOperationException">The Mat is empty, or not a 2-D 8-bit image.</exception>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public unsafe byte this[int row, int col, int channel]
        {
            get
            {
                var cache = Cache ?? throw new InvalidOperationException("The Mat is empty.");
                var layout = cache.Layout;
                long[] shape = layout.Shape;
                long[] strides = layout.ByteStrides;
                int channels = layout.Type.Channels();
                if (layout.Type.Depth() != MatDepth.CV_8U || shape.Length != (channels > 1 ? 3 : 2))
                    throw new InvalidOperationException($"The byte indexer reads 2-D 8-bit images; this Mat is {layout.Type} with shape ({string.Join(", ", shape)}). Read it through data.");
                if ((ulong)row >= (ulong)shape[0] || (ulong)col >= (ulong)shape[1] || (uint)channel >= (uint)channels)
                    throw new IndexOutOfRangeException($"Pixel ({row}, {col}, {channel}) is outside the ({string.Join(", ", shape)}) image.");

                // The cached view's lease keeps these pixels alive while the cache is reachable.
                long offset = row * strides[0] + col * strides[1] + (channels > 1 ? channel * strides[2] : 0);
                byte value = layout.Data[offset];
                GC.KeepAlive(cache);
                return value;
            }
        }
    }
}
