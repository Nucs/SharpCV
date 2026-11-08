using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace SharpCV
{
    /// <summary>
    ///     <c>cv2.VideoCapture</c>: frames from a video file, a stream or a camera. Dispose it to close the source (the
    ///     finalizer closes it otherwise).
    /// </summary>
    public class VideoCapture : DisposableObject
    {
        /// <summary>Wraps a <c>cv::VideoCapture*</c> and takes ownership of it.</summary>
        /// <param name="handle">The capture created by OpenCvSharpExtern.</param>
        public VideoCapture(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary>The native capture.</summary>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        private IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(VideoCapture));

        /// <summary><c>cap.read()</c>: the next frame into a NEW Mat.</summary>
        /// <returns>(retval, image) as cv2 returns them; at the end of the stream retval is false and the image empty.</returns>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        /// <exception cref="OpenCVException">The backend failed.</exception>
        public (bool, Mat) read()
        {
            var dst = new Mat();
            return (read(dst).Item1, dst);
        }

        /// <summary>
        ///     <c>cap.read(image)</c>: the next frame INTO <paramref name="image"/>, which OpenCV reuses when the frame fits
        ///     (no allocation per frame) and reallocates otherwise.
        /// </summary>
        /// <param name="image">The frame buffer.</param>
        /// <returns>(retval, image): whether a frame was read, and <paramref name="image"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The capture or the image was disposed.</exception>
        /// <exception cref="OpenCVException">The backend failed, or the Mat views a read-only NDArray.</exception>
        public (bool, Mat) read(Mat image)
        {
            if (image is null)
                throw new ArgumentNullException(nameof(image));
            var handle = Handle;
            int output = 0;
            // read() takes the cv::Mat itself, not an _OutputArray, so the cached shape is reset by hand.
            var frame = image.OutputArray;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_read_OutputArray(handle, frame, out output));
            GC.KeepAlive(this);
            // OpenCV may have reallocated the frame (another size): report the buffer the Mat holds now.
            image.TrackNativeMemory();
            GC.KeepAlive(image);
            return (output != 0, image);
        }

        /// <summary>
        ///     <c>cap.read(image)</c> into an NDArray frame buffer, the zero-allocation loop NumSharp code wants: OpenCV
        ///     decodes straight into the array's memory when its shape and dtype fit the frame. When they do not, OpenCV
        ///     allocates a new frame and the result is a view of it, as cv2 returns a new array (the given one is left
        ///     untouched).
        /// </summary>
        /// <param name="image">A writeable frame buffer, e.g. <c>np.empty((720, 1280, 3), np.uint8)</c>.</param>
        /// <returns>(retval, image): whether a frame was read, and <paramref name="image"/> itself when the frame went
        ///     into it, otherwise the new frame.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        /// <exception cref="OpenCVException">The array is read-only, its dtype has no depth, its layout would need a
        ///     copy, or the backend failed.</exception>
        public unsafe (bool, NDArray) read(NDArray image)
        {
            using var mat = Mat.ForOutput(image, nameof(image));
            void* before = image.data.Pointer;
            var (ok, _) = read(mat);
            if (!ok)
                return (false, image);

            // Same data pointer: the frame was decoded into the caller's memory. Otherwise OpenCV reallocated the Mat.
            using var after = mat.AsNDArray(MatCodecMode.View);
            // `is not null`: NDArray overloads != element-wise.
            if (after is not null && after.data.Pointer == before)
                return (true, image);
            return (true, after?.view());
        }

        /// <summary><c>cap.isOpened()</c>: whether the source opened.</summary>
        /// <returns>True when frames can be read.</returns>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        public bool isOpened()
        {
            var handle = Handle;
            int output = 0;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_isOpened(handle, out output));
            GC.KeepAlive(this);
            return output != 0;
        }

        /// <summary><c>cap.get(propId)</c>: a capture property (frame size, count, FPS, position, ...).</summary>
        /// <param name="propId">The property.</param>
        /// <returns>Its value; 0 when the backend does not support it.</returns>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        public double get(VideoCaptureProperties propId)
        {
            var handle = Handle;
            double output = 0;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_get(handle, (int)propId, out output));
            GC.KeepAlive(this);
            return output;
        }

        /// <summary><c>cap.release()</c>: closes the source now; the object stays usable and closed.</summary>
        /// <exception cref="ObjectDisposedException">The capture was disposed.</exception>
        public void release()
        {
            var handle = Handle;
            NativeMethods.Call(() => cv2_native_api.videoio_VideoCapture_release(handle));
            GC.KeepAlive(this);
        }

        /// <summary>Deletes the native capture (closing the source). Until SharpCV 0.20 this was the base class's
        ///     <c>Marshal.FreeHGlobal</c>, which corrupts the C++ heap.</summary>
        protected override void FreeHandle()
        {
            if (_handle != IntPtr.Zero)
            {
                try
                {
                    cv2_native_api.videoio_VideoCapture_delete(_handle);
                }
                catch (System.Runtime.InteropServices.SEHException)
                {
                    // Never let a native surprise escape a finalizer.
                }
            }
        }
    }
}
