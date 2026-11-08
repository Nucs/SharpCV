using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary>
        ///     <c>cv2.imread(filename, flags)</c>: decodes an image file. Like OpenCV (and unlike most .NET readers) a
        ///     missing or undecodable file gives an EMPTY Mat, whose <see cref="Mat.data"/> is null (cv2 returns
        ///     <c>None</c>), not an exception.
        /// </summary>
        /// <param name="filename">The file.</param>
        /// <param name="flags">Colour (BGR, the default), grayscale, unchanged, ...</param>
        /// <returns>The image (<c>(H, W, 3)</c> uint8 for a colour image).</returns>
        /// <exception cref="OpenCVException">The decoder failed hard (OpenCV error, not a missing file).</exception>
        public Mat imread(string filename, IMREAD_COLOR flags = IMREAD_COLOR.IMREAD_COLOR)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.imgcodecs_imread(filename, (int)flags, out handle));
            return new Mat(handle).Tracked();
        }

        /// <summary><c>cv2.imwrite(filename, img, params)</c>: encodes by the file extension and writes the file.</summary>
        /// <param name="filename">The output file.</param>
        /// <param name="img">The image (a Mat or an NDArray).</param>
        /// <param name="params">Encoder parameter pairs (id, value).</param>
        /// <returns>True when the file was written.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">No encoder for the extension, the image type does not fit it, or cv2
        ///     refuses the NDArray.</exception>
        public bool imwrite(string filename, InputArray img, params int[] @params)
        {
            using var mat = img.Open("img");
            // imgcodecs_imwrite takes the cv::Mat itself; Handle refuses a disposed Mat instead of passing OpenCV null.
            var handle = mat.Mat.Handle;
            @params ??= Array.Empty<int>();
            int output = 0;
            NativeMethods.Call(() => cv2_native_api.imgcodecs_imwrite(filename, handle, @params, @params.Length, out output));
            return output != 0;
        }

        /// <summary>A region of <paramref name="src"/> that SHARES its pixels (an alias of the ROI indexer).</summary>
        /// <param name="src">The image.</param>
        /// <param name="rect">The region (x, y, width, height).</param>
        /// <returns>A new Mat header over the region.</returns>
        /// <exception cref="OpenCVException">The region does not lie inside the image.</exception>
        public Mat imcrop(Mat src, Rect rect)
        {
            IntPtr source = src;
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.core_Mat_new7(source, rect, out handle));
            var roi = new Mat(handle);
            roi.InheritSource(src);
            GC.KeepAlive(src);
            return roi;
        }

        /// <summary>
        ///     <c>cv2.imdecode(buf, flags)</c>: decodes an image held in memory. <paramref name="buf"/> is any array OpenCV
        ///     accepts as encoded bytes (cv2 asserts a continuous single-channel uint8 array); a contiguous NumSharp
        ///     <c>uint8</c> NDArray is read WITHOUT a copy.
        /// </summary>
        /// <param name="buf">The encoded bytes (a Mat or an NDArray).</param>
        /// <param name="flags">The decode flags.</param>
        /// <returns>The image; empty (data null) when the bytes are no image, as cv2 returns <c>None</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="buf"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The buffer is not 8-bit single-channel, the decoder failed, or cv2 refuses
        ///     the NDArray.</exception>
        public Mat imdecode(InputArray buf, IMREAD_COLOR flags)
        {
            using var bytes = buf.Open("buf");
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.imgcodecs_imdecode_InputArray(bytes.Mat.InputArray, (int)flags, out handle));
            return new Mat(handle).Tracked();
        }

        /// <summary>
        ///     <c>cv2.imencode(ext, img, params)</c>: compresses the image in memory. The bytes come back as a 1-D uint8
        ///     NDArray that VIEWS OpenCV's output buffer (no copy); the buffer is freed when the last view of it is disposed
        ///     or collected.
        /// </summary>
        /// <param name="ext">The format extension (".png", ".jpg", ".webp", ...).</param>
        /// <param name="img">The image (a Mat or an NDArray).</param>
        /// <param name="params">Encoder parameter pairs (id, value).</param>
        /// <returns>(retval, buf) as cv2 returns them: whether encoding succeeded, and the encoded bytes.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ext"/> or <paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">No encoder for the extension, the image type does not fit it, or cv2
        ///     refuses the NDArray.</exception>
        public unsafe (bool, NDArray) imencode(string ext, InputArray img, params int[] @params)
        {
            if (string.IsNullOrEmpty(ext))
                throw new ArgumentNullException(nameof(ext));
            @params ??= Array.Empty<int>();

            // Opened before the vector exists, so a refused argument has nothing native to clean up.
            using var src = img.Open("img");
            IntPtr vector = cv2_native_api.vector_uchar_new1();
            int ok = 0;
            try
            {
                NativeMethods.Call(() => cv2_native_api.imgcodecs_imencode_vector(ext, src.Mat.InputArray, vector, @params, @params.Length, out ok));
            }
            catch
            {
                cv2_native_api.vector_uchar_delete(vector);
                throw;
            }

            long count = (long)cv2_native_api.vector_uchar_getSize(vector);
            if (count == 0)
            {
                // getPointer would throw std::out_of_range across the boundary for an empty vector.
                cv2_native_api.vector_uchar_delete(vector);
                return (ok != 0, np.empty(new Shape(0), np.uint8));
            }

            byte* bytes = (byte*)cv2_native_api.vector_uchar_getPointer(vector);
            int released = 0;
            void Release()
            {
                // NumSharp's last-reference hook (Dispose or finalizer): free OpenCV's buffer exactly once.
                if (System.Threading.Interlocked.Exchange(ref released, 1) == 0)
                    cv2_native_api.vector_uchar_delete(vector);
            }

            var slice = new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(bytes, count, Release));
            try
            {
                return (ok != 0, np.ndarray(new Shape(count), np.uint8, buffer: slice));
            }
            catch
            {
                Release();
                throw;
            }
        }

        /// <summary>
        /// Reads image from the specified buffer in memory.
        /// </summary>
        /// <param name="span">The input slice of bytes.</param>
        /// <param name="flags">The same flags as in imread</param>
        /// <returns>The image; empty when the bytes are no image.</returns>
        /// <exception cref="ArgumentException"><paramref name="span"/> is empty.</exception>
        /// <exception cref="OpenCVException">The decoder failed.</exception>
        public static Mat ImDecode(byte[] span, IMREAD_COLOR flags = IMREAD_COLOR.IMREAD_COLOR)
        {
            if (span.Length == 0)
                throw new ArgumentException("Empty span", nameof(span));

            unsafe
            {
                fixed (byte* pBuf = span)
                {
                    // The buffer stays pinned for the whole native call: Call runs the lambda synchronously inside it.
                    // A fixed local cannot be captured by a lambda, so the address crosses as an IntPtr.
                    IntPtr address = (IntPtr)pBuf;
                    IntPtr ret = IntPtr.Zero;
                    NativeMethods.Call(() => cv2_native_api.imgcodecs_imdecode_vector((byte*)address, span.Length, flags, out ret));
                    return new Mat(ret).Tracked();
                }
            }
        }

        /// <summary>
        /// Compresses the image and stores it in the memory buffer
        /// </summary>
        /// <param name="ext">The file extension that defines the output format</param>
        /// <param name="img">The image to be written (a Mat or an NDArray)</param>
        /// <param name="buf">Output buffer resized to fit the compressed image.</param>
        /// <param name="prms">Format-specific parameters.</param>
        /// <returns>True when encoding succeeded.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ext"/> or <paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">No encoder for the extension, the image type does not fit it, or cv2
        ///     refuses the NDArray.</exception>
        public static bool ImEncode(string ext, InputArray img, out byte[] buf, int[]? prms = null)
        {
            if (string.IsNullOrEmpty(ext))
                throw new ArgumentNullException(nameof(ext));
            using var src = img.Open("img");
            if (prms == null)
                prms = Array.Empty<int>();
			//img.ThrowIfDisposed();

			using var bufVec = new VectorOfByte();
            int ret = 0;
            // The image is the encoder's INPUT; SharpCV passed its OutputArray here until 0.20.
            NativeMethods.Call(() => cv2_native_api.imgcodecs_imencode_vector(ext, src.Mat.InputArray, bufVec.Handle, prms, prms.Length, out ret));

			buf = bufVec.ToArray();

			return ret != 0;
        }

    }
}
