using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     The <c>cv2</c> module (reach it as <c>Binding.cv2</c> with <c>using static SharpCV.Binding;</c>): OpenCV's
    ///     functions with opencv-python's names, argument order and defaults. An image a function reads is an
    ///     <see cref="InputArray"/> and an image it draws into an <see cref="InputOutputArray"/>: pass a <see cref="Mat"/>
    ///     or a NumSharp NDArray to either.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     An NDArray argument is converted by cv2's rules for the one call (shared when its layout fits cv::Mat, copied
    ///     otherwise) and the conversion is released when the call returns, so passing NDArrays in a loop leaves nothing
    ///     for the finalizer.
    ///     </para>
    ///     <para>
    ///     Every function raises <see cref="OpenCVException"/> (cv2's <c>cv2.error</c>, same message) when OpenCV rejects
    ///     the call; none returns an empty result for bad input any more.
    ///     </para>
    ///     <para>
    ///     <b>The pre-0.15 overloads</b> (<c>cvtColor(Mat ...)</c>, <c>rectangle(NDArray ...)</c>, and so on, in
    ///     cv2.compat.cs) are kept so code written against them still compiles: an <c>IntPtr</c> argument (which converts
    ///     to an OWNING <see cref="Mat"/>), a <c>Mat</c>-typed method group, a C# 12 caller. Each forwards to its proxy
    ///     twin and carries <c>[OverloadResolutionPriority(-1)]</c>, so a C# 13+ compiler (the default from net9.0 on, or
    ///     <c>LangVersion</c> 13+) binds Mat and NDArray arguments to the proxy overloads. A C# 12 compiler (the default
    ///     for net8.0 projects) ignores the attribute and binds an NDArray argument to the <c>Mat</c> overload, the better
    ///     conversion target: its <c>implicit operator Mat(NDArray)</c> temporary then waits for the finalizer, kept
    ///     bounded by SharpCV's native-memory tracker. A net8.0 project gets the proxies with
    ///     <c>&lt;LangVersion&gt;latest&lt;/LangVersion&gt;</c>. The drawing functions keep their NDArray overloads, so a
    ///     C# 12 caller draws into an NDArray in place either way.
    ///     </para>
    /// </remarks>
    public partial class cv_api
    {
        /// <summary><c>cv2.flip(src, flipCode)</c>: mirrors the image.</summary>
        /// <param name="img">The image (a Mat or an NDArray).</param>
        /// <param name="mode">Vertical (0), horizontal (1) or both (-1).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input (an n-D Mat), or cv2 refuses the NDArray (its
        ///     dtype has no OpenCV depth).</exception>
        public Mat flip(InputArray img, FLIP_MODE mode)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_flip(src.Mat.InputArray, dst.OutputArray, (int)mode));
            return dst.Tracked();
        }

        /// <summary><c>cv2.rotate(src, rotateCode)</c>: rotates by a multiple of 90 degrees (no interpolation).</summary>
        /// <param name="img">The image (a Mat or an NDArray).</param>
        /// <param name="flags">The rotation.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Mat rotate(InputArray img, RotateFlags flags)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_rotate(src.Mat.InputArray, dst.OutputArray, (int)flags));
            return dst.Tracked();
        }

        /// <summary><c>cv2.split(m)</c>: one single-channel image per channel, each a COPY (a NumSharp channel view
        ///     <c>img.data[":, :, 0"]</c> shares memory instead).</summary>
        /// <param name="img">The multi-channel image (a Mat or an NDArray).</param>
        /// <returns>The planes.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Mat[] split(InputArray img)
        {
            using var m = img.Open("m");
            using (var vec = new VectorOfMat())
            {
                // core_split takes the cv::Mat itself, not an _InputArray.
                NativeMethods.Call(() => cv2_native_api.core_split(m.Mat, vec));
                return vec.ToArray();
            }
        }

        /// <summary><c>cv2.hconcat(src)</c>: the images side by side (same height and type).</summary>
        /// <param name="imgs">The images, Mats and NDArrays mixed as needed.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imgs"/> or one of its images is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The heights or types differ, or cv2 refuses an NDArray.</exception>
        public Mat hconcat(params InputArray[] imgs)
            => Concatenate(imgs, horizontal: true);

        /// <summary><c>cv2.vconcat(src)</c>: the images stacked vertically (same width and type).</summary>
        /// <param name="imgs">The images, Mats and NDArrays mixed as needed.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imgs"/> or one of its images is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The widths or types differ, or cv2 refuses an NDArray.</exception>
        public Mat vconcat(params InputArray[] imgs)
            => Concatenate(imgs, horizontal: false);

        /// <summary>
        ///     hconcat / vconcat over proxies: opens every image for the call (NDArrays become Mats) and releases the
        ///     temporaries afterwards, also when an image is refused half-way through.
        /// </summary>
        /// <param name="imgs">The images.</param>
        /// <param name="horizontal">True for hconcat.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imgs"/> or one of its images is null.</exception>
        /// <exception cref="ObjectDisposedException">One of the Mats was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the images, or cv2 refuses an NDArray.</exception>
        private static Mat Concatenate(InputArray[] imgs, bool horizontal)
        {
            if (imgs is null)
                throw new ArgumentNullException(nameof(imgs));

            var args = new MatArgument[imgs.Length];
            int opened = 0;
            try
            {
                for (; opened < imgs.Length; opened++)
                    args[opened] = imgs[opened].Open("src");

                var handles = new IntPtr[args.Length];
                for (int i = 0; i < args.Length; i++)
                    handles[i] = args[i].Mat.Handle;
                return Concatenate(handles, horizontal);
            }
            finally
            {
                // Only the arguments opened so far: a refusal at image k leaves k temporaries to release.
                for (int i = 0; i < opened; i++)
                    args[i].Dispose();
            }
        }

        /// <summary>hconcat / vconcat over Mats the caller owns.</summary>
        /// <param name="imgs">The images.</param>
        /// <param name="horizontal">True for hconcat.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="imgs"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">One of the images was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the images.</exception>
        private static Mat Concatenate(Mat[] imgs, bool horizontal)
        {
            if (imgs is null)
                throw new ArgumentNullException(nameof(imgs));
            var handles = new IntPtr[imgs.Length];
            for (int i = 0; i < imgs.Length; i++)
            {
                // OpenCvSharpExtern dereferences every cv::Mat* it is given: a null or disposed element (IntPtr.Zero until
                // SharpCV 0.15) is an access violation, which .NET cannot catch, so both are refused here instead.
                if (imgs[i] is null)
                    throw new ArgumentNullException(nameof(imgs), $"imgs[{i}] is null.");
                handles[i] = imgs[i].Handle;
            }

            var result = Concatenate(handles, horizontal);
            GC.KeepAlive(imgs);
            return result;
        }

        /// <summary>The native hconcat / vconcat over <c>cv::Mat*</c> handles the caller keeps alive.</summary>
        /// <param name="handles">The images' headers.</param>
        /// <param name="horizontal">True for hconcat.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">OpenCV rejects the images.</exception>
        private static Mat Concatenate(IntPtr[] handles, bool horizontal)
        {
            var dst = new Mat();
            if (horizontal)
                NativeMethods.Call(() => cv2_native_api.core_hconcat1(handles, (uint)handles.Length, dst.OutputArray));
            else
                NativeMethods.Call(() => cv2_native_api.core_vconcat1(handles, (uint)handles.Length, dst.OutputArray));
            return dst.Tracked();
        }
    }
}
