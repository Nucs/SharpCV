using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace SharpCV
{
    public partial class Mat
    {
        /// <summary>
        ///     An empty Mat (no pixels), the destination OpenCV functions allocate their result into.
        /// </summary>
        /// <exception cref="OpenCVException">OpenCV failed to allocate the header.</exception>
        /// <exception cref="DllNotFoundException">No OpenCvSharpExtern runtime package is installed.</exception>
        public Mat()
        {
            // Every wrapper call makes one of these for its result: a static lambda over the handle slot allocates
            // nothing, where a closure cost two objects per call.
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(ref handle, static (ref IntPtr h) => cv2_native_api.core_Mat_new1(out h));
            _handle = handle;
        }

        /// <summary>
        ///     Wraps an existing cv::Mat pointer and TAKES OWNERSHIP of it: the Mat deletes it when disposed or collected,
        ///     so never wrap the same pointer twice.
        /// </summary>
        /// <param name="handle">A <c>cv::Mat*</c> created by OpenCvSharpExtern.</param>
        /// <exception cref="NotSupportedException">The loaded OpenCvSharpExtern is not the verified OpenCV 4.13.</exception>
        public Mat(IntPtr handle)
        {
            // A handle may come from outside SharpCV, before any SharpCV call registered the error callback and checked
            // the library version; the field getters this Mat calls next skip that check.
            if (handle != IntPtr.Zero)
                NativeMethods.Initialize();
            _handle = handle;
        }

        /// <summary>
        ///     A Mat over an NDArray, converted by opencv-python's rules (a port of its <c>pyopencv_to</c>): it shares
        ///     <paramref name="nd"/>'s memory when the layout fits cv::Mat, so OpenCV's writes reach the array.
        /// </summary>
        /// <param name="nd">
        ///     The array. A 3-D array whose last axis has at most 512 entries becomes a multi-channel 2-D Mat
        ///     (<c>(H, W, 3)</c> uint8 is a BGR image); any other shape becomes an n-D Mat, a 1-D array an N x 1 Mat.
        /// </param>
        /// <param name="mode">
        ///     <see cref="MatCodecMode.Auto"/> (default) shares when it can and copies otherwise (a transposed, reversed,
        ///     column-strided or broadcast array, or int64 / uint64 cast to int32), exactly as cv2 treats an input array;
        ///     <see cref="MatCodecMode.View"/> refuses to copy; <see cref="MatCodecMode.Copy"/> always copies.
        /// </param>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">The dtype has no OpenCV depth (bool, uint32, char, decimal, complex), the
        ///     array has 32 or more dimensions, or <see cref="MatCodecMode.View"/> met a layout that needs a copy.</exception>
        /// <remarks>
        ///     The Mat keeps a counted NumSharp view of <paramref name="nd"/> (or owns the copy), so the caller may dispose
        ///     <paramref name="nd"/> right away. A read-only array (a broadcast view) can be read but never written: the Mat
        ///     refuses to serve as an OpenCV output.
        /// </remarks>
        public Mat(NDArray nd, MatCodecMode mode = MatCodecMode.Auto)
            : this(nd, mode, "array", output: false, ndMat: false, borrow: false)
        {
        }

        /// <summary>
        ///     The conversion every NDArray entry point shares, with cv2's per-argument flags.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="mode">Share, refuse to copy, or copy.</param>
        /// <param name="name">The argument name used in refusal messages.</param>
        /// <param name="output">True when OpenCV writes into the array (it must be writeable and viewable).</param>
        /// <param name="ndMat">True for an n-D blob argument: no multi-channel folding.</param>
        /// <param name="borrow">True for a Mat that lives for one call (<see cref="ForCall"/>, <see cref="ForOutput"/>):
        ///     a viewable array is referenced as it is instead of through a counted NumSharp view, and never disposed by
        ///     this Mat. A borrowing Mat must not outlive the caller's own reference to <paramref name="nd"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">A cv2 conversion refusal, or OpenCV failed to build the header.</exception>
        internal Mat(NDArray nd, MatCodecMode mode, string name, bool output, bool ndMat, bool borrow)
        {
            Build(nd, mode, name, output, ndMat, borrow);
        }

        /// <summary>
        ///     The NDArray conversion's work, shared by the constructor and a deferred Mat's first native use
        ///     (<see cref="Materialize"/>): plans the conversion, builds the cv::Mat header over the planned memory, and
        ///     records what the Mat keeps (the view or copy, its read-only flag, the copy's memory report).
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="mode">Share, refuse to copy, or copy.</param>
        /// <param name="name">The argument name used in refusal messages.</param>
        /// <param name="output">True when OpenCV writes into the array.</param>
        /// <param name="ndMat">True for an n-D blob argument: no multi-channel folding.</param>
        /// <param name="borrow">True to reference a viewable array as it is (see the constructor).</param>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">A cv2 conversion refusal, or OpenCV failed to build the header.</exception>
        private void Build(NDArray nd, MatCodecMode mode, string name, bool output, bool ndMat, bool borrow)
        {
            // The plan owns its view or copy until TakeSource; if the native header fails, disposing the plan releases it.
            using var plan = NDArrayToMat.Plan(nd, mode, name, output, ndMat, borrow);
            // The plan travels as the state of a static lambda: no closure for the call every NDArray argument makes.
            var call = (Plan: plan, Handle: IntPtr.Zero);
            NativeMethods.Call(ref call, static (ref (NDArrayToMat Plan, IntPtr Handle) s) =>
                cv2_native_api.core_Mat_new9(s.Plan.Dims, s.Plan.Sizes, s.Plan.Type, s.Plan.Data, s.Plan.Steps, out s.Handle));

            _handle = call.Handle;
            _sourceBorrowed = plan.Borrowed;
            _sourceReadOnly = !plan.Source.Shape.IsWriteable;
            _sourceData = plan.Data;
            if (plan.Copied)
            {
                // A copy only this Mat references: count it until the Mat dies, so temporaries left to the finalizer
                // cannot pile up unseen.
                long bytes = plan.Source.size * plan.Source.itemsize;
                if (bytes >= PressureThreshold)
                {
                    _copyPressure = bytes;
                    NativeMemoryTracker.Add(bytes);
                }
            }

            _source = plan.TakeSource();
        }

        /// <summary>
        ///     The Mat for an NDArray that OpenCV writes INTO (a drawing target, a frame buffer): always a view of a
        ///     writeable array, as cv2 demands of output arguments.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="name">The argument name used in refusal messages.</param>
        /// <returns>The Mat; dispose it right after the call.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">The array is read-only ("marked as output argument, but provided NumPy
        ///     array marked as readonly"), its dtype has no depth, or its layout would need a copy ("Layout of the output
        ///     array img is incompatible with cv::Mat").</exception>
        /// <remarks>Lives for one call, so it borrows <paramref name="nd"/> instead of taking a counted view.</remarks>
        internal static Mat ForOutput(NDArray nd, string name)
            => new Mat(nd, MatCodecMode.View, name, output: true, ndMat: false, borrow: true);

        /// <summary>
        ///     The Mat for an NDArray that one OpenCV call READS (an <see cref="InputArray"/> argument): shared when the
        ///     layout fits, copied otherwise, as cv2 converts an input array, and disposed when the call returns.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="name">The argument name used in refusal messages.</param>
        /// <param name="ndMat">True for an n-D blob argument: no multi-channel folding.</param>
        /// <returns>The Mat; dispose it right after the call.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">The dtype has no depth, or the array has too many dimensions.</exception>
        /// <remarks>Borrows <paramref name="nd"/> when it is viewable (no counted view): the caller holds the array
        ///     for the length of the call anyway.</remarks>
        internal static Mat ForCall(NDArray nd, string name, bool ndMat)
            => new Mat(nd, MatCodecMode.Auto, name, output: false, ndMat, borrow: true);

        /// <summary>
        ///     The Mat <c>implicit operator Mat(NDArray)</c> returns: a DEFERRED Mat that holds a counted view of
        ///     <paramref name="nd"/> and builds its native header (and any copy) only when something first needs it. Until
        ///     then a cv2 function given this Mat (a C# 12 caller bound to a pre-0.15 Mat overload, or an InputArray built
        ///     from it) reads the ARRAY instead, converted for that one call and released when it returns, so an NDArray
        ///     argument costs the same whichever overload the caller's compiler picked.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The deferred Mat.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="OpenCVException">The conversion refuses the array (its dtype has no depth, it has 32 or more
        ///     dimensions, an axis is too long): raised here, as the eager conversion raised it.</exception>
        /// <remarks>
        ///     Transparent by design: any native use (<see cref="data"/>, <see cref="shape"/>, an ROI, an output argument,
        ///     <c>(IntPtr)mat</c>) builds exactly the Mat <c>new Mat(nd)</c> builds, with the counted view taken here, so
        ///     the caller may still dispose <paramref name="nd"/> at once. Its finalizer is off until then: the view has
        ///     its own, and no native state exists yet.
        /// </remarks>
        internal static Mat Deferred(NDArray nd)
        {
            NDArrayToMat.ValidateInput(nd, "array", ndMat: false);
            var mat = new Mat(IntPtr.Zero) { _pending = nd.view() };
            GC.SuppressFinalize(mat);
            return mat;
        }

        //this method copies Mat to a new NDArray
        /// <summary>The pixels as an independent C-contiguous copy (<see cref="AsNDArray"/> with
        ///     <see cref="MatCodecMode.Copy"/>).</summary>
        /// <returns>The copy, or null for an empty Mat.</returns>
        private unsafe NDArray CopyToNDArray()
        {
            return AsNDArray(MatCodecMode.Copy);
        }

        //this method wraps without copying Mat.
        /// <summary>The pixels as a zero-copy view (<see cref="AsNDArray"/> with <see cref="MatCodecMode.View"/>).</summary>
        /// <returns>The view, or null for an empty Mat.</returns>
        private unsafe NDArray WrapWithNDArray()
        {
            return AsNDArray(MatCodecMode.View);
        }
    }
}
