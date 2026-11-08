using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace SharpCV
{
    /// <summary><c>cv2.dnn.Net</c>: a loaded network. Set its input, then run <see cref="forward"/>.</summary>
    public class Net : DisposableObject
    {
        /// <summary>An empty network.</summary>
        /// <exception cref="OpenCVException">OpenCV failed to create it.</exception>
        public Net()
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_Net_new(out handle));
            _handle = handle;
        }

        /// <summary>Wraps a <c>cv::dnn::Net*</c> and takes ownership of it.</summary>
        /// <param name="handle">The network created by OpenCvSharpExtern.</param>
        public Net(IntPtr handle)
        {
            _handle = handle;
        }

        /// <summary><c>net.setInput(blob, name)</c>. OpenCV copies the blob, so it may be released afterwards.</summary>
        /// <param name="blob">The input (an NCHW blob from <see cref="Dnn.blobFromImage"/>).</param>
        /// <param name="name">The input layer; empty for the first.</param>
        /// <exception cref="OpenCVException">OpenCV rejects the blob.</exception>
        public void setInput(Mat blob, string name = "")
        {
            var handle = _handle;
            NativeMethods.Call(() => cv2_native_api.dnn_Net_setInput(handle, blob, name));
            GC.KeepAlive(blob);
            GC.KeepAlive(this);
        }

        /// <summary>
        ///     <c>net.setInput(blob, name)</c> for an NDArray blob: an n-D argument, as cv2 marks it (<c>CV_ND</c>), so a
        ///     3-D array stays three dimensions instead of becoming a multi-channel image. A C-contiguous float32 array
        ///     crosses without a copy (OpenCV then copies it into the input layer).
        /// </summary>
        /// <param name="blob">The input, e.g. <c>(1, 3, 224, 224)</c> float32.</param>
        /// <param name="name">The input layer; empty for the first.</param>
        /// <exception cref="ArgumentNullException"><paramref name="blob"/> is null.</exception>
        /// <exception cref="OpenCVException">The dtype has no OpenCV depth, or OpenCV rejects the blob.</exception>
        public void setInput(NDArray blob, string name = "")
        {
            using var mat = new Mat(blob, MatCodecMode.Auto, nameof(blob), output: false, ndMat: true);
            setInput(mat, name);
        }

        /// <summary>
        ///     <c>net.forward(outputName)</c>: runs the network.
        /// </summary>
        /// <param name="outputName">The output layer; empty for the last.</param>
        /// <returns>A new Mat with the output, independent of the network: OpenCV returns a header over the network's
        ///     own buffer, which the next forward pass overwrites, so it is cloned (cv2's numpy result is a copy
        ///     too).</returns>
        /// <exception cref="OpenCVException">No input was set, or a layer failed.</exception>
        public Mat forward(string outputName = "")
        {
            var handle = _handle;
            IntPtr output = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.dnn_Net_forward1(handle, outputName, out output));
            GC.KeepAlive(this);
            using var shared = new Mat(output);
            IntPtr clone = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.core_Mat_clone(output, out clone));
            return new Mat(clone).Tracked();
        }

        /// <summary>Deletes the native network.</summary>
        protected override void FreeHandle()
        {
            if (_handle != IntPtr.Zero)
                cv2_native_api.dnn_Net_delete(_handle);
        }

        /// <summary>The native <c>cv::dnn::Net*</c>.</summary>
        /// <param name="net">The network.</param>
        /// <returns>Its handle.</returns>
        public static implicit operator IntPtr(Net net)
            => net._handle;

        /// <summary>Wraps a <c>cv::dnn::Net*</c> and takes ownership of it.</summary>
        /// <param name="handle">The pointer.</param>
        /// <returns>The owning network.</returns>
        public static implicit operator Net(IntPtr handle)
            => new Net(handle);
    }
}
