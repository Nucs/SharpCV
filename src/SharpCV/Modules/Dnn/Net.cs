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

        /// <summary>The native network.</summary>
        /// <exception cref="ObjectDisposedException">The network was disposed: OpenCvSharpExtern would dereference a null
        ///     <c>cv::dnn::Net*</c>, an access violation .NET cannot catch.</exception>
        private IntPtr Handle => _handle != IntPtr.Zero ? _handle : throw new ObjectDisposedException(nameof(Net));

        /// <summary>
        ///     <c>net.setInput(blob, name)</c>. OpenCV copies the blob into the input layer, so it may be released
        ///     afterwards. An NDArray blob is an n-D argument, as cv2 marks it (<c>CV_ND</c>): a 3-D array stays three
        ///     dimensions instead of becoming a multi-channel image, and a C-contiguous float32 array crosses without a
        ///     copy.
        /// </summary>
        /// <param name="blob">The input: an NCHW blob Mat from <see cref="Dnn.blobFromImage(InputArray, double, System.ValueTuple{int, int}, System.Nullable{Scalar}, bool, bool)"/>, or an NDArray such as a
        ///     <c>(1, 3, 224, 224)</c> float32 array.</param>
        /// <param name="name">The input layer; empty for the first.</param>
        /// <exception cref="ArgumentNullException"><paramref name="blob"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">The network or a Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the blob, or cv2 refuses the NDArray (its dtype has no
        ///     depth).</exception>
        public void setInput(InputArray blob, string name = "")
        {
            using var input = blob.Open("blob", ndMat: true);
            // dnn_Net_setInput takes the cv::Mat itself; Handle refuses a disposed Mat instead of passing OpenCV null.
            var source = input.Mat.Handle;
            var handle = Handle;
            NativeMethods.Call(() => cv2_native_api.dnn_Net_setInput(handle, source, name));
            GC.KeepAlive(this);
        }

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for a Mat argument.</summary>
        /// <inheritdoc cref="setInput(InputArray, string)"/>
        [System.Runtime.CompilerServices.OverloadResolutionPriority(-1)]
        public void setInput(Mat blob, string name = "")
            => setInput((InputArray)blob, name);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputArray overload, so
        ///     the blob stays n-dimensional (cv2's <c>CV_ND</c>) and is released when the call returns, whatever the
        ///     caller's C# version.</summary>
        /// <inheritdoc cref="setInput(InputArray, string)"/>
        [System.Runtime.CompilerServices.OverloadResolutionPriority(-1)]
        public void setInput(NDArray blob, string name = "")
            => setInput((InputArray)blob, name);

        /// <summary>
        ///     <c>net.forward(outputName)</c>: runs the network.
        /// </summary>
        /// <param name="outputName">The output layer; empty for the last.</param>
        /// <returns>A new Mat with the output, independent of the network: OpenCV returns a header over the network's
        ///     own buffer, which the next forward pass overwrites, so it is cloned (cv2's numpy result is a copy
        ///     too).</returns>
        /// <exception cref="ObjectDisposedException">The network was disposed.</exception>
        /// <exception cref="OpenCVException">No input was set, or a layer failed.</exception>
        public Mat forward(string outputName = "")
        {
            var handle = Handle;
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
