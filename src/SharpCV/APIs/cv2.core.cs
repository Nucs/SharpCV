using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     The <c>cv2</c> module (reach it as <c>Binding.cv2</c> with <c>using static SharpCV.Binding;</c>): OpenCV's
    ///     functions with opencv-python's names, argument order and defaults. Every image parameter is a
    ///     <see cref="Mat"/>, and an NDArray converts to one implicitly by cv2's rules.
    /// </summary>
    /// <remarks>
    ///     Every function raises <see cref="OpenCVException"/> (cv2's <c>cv2.error</c>, same message) when OpenCV rejects
    ///     the call; none returns an empty result for bad input any more.
    /// </remarks>
    public partial class cv_api
    {
        /// <summary><c>cv2.flip(src, flipCode)</c>: mirrors the image.</summary>
        /// <param name="img">The image.</param>
        /// <param name="mode">Vertical (0), horizontal (1) or both (-1).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">OpenCV rejects the input (an n-D Mat).</exception>
        public Mat flip(Mat img, FLIP_MODE mode)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_flip(img.InputArray, dst.OutputArray, (int)mode));
            GC.KeepAlive(img);
            return dst;
        }

        /// <summary><c>cv2.rotate(src, rotateCode)</c>: rotates by a multiple of 90 degrees (no interpolation).</summary>
        /// <param name="img">The image.</param>
        /// <param name="flags">The rotation.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">OpenCV rejects the input.</exception>
        public Mat rotate(Mat img, RotateFlags flags)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_rotate(img.InputArray, dst.OutputArray, (int)flags));
            GC.KeepAlive(img);
            return dst;
        }

        /// <summary><c>cv2.split(m)</c>: one single-channel image per channel, each a COPY (a NumSharp channel view
        ///     <c>img.data[":, :, 0"]</c> shares memory instead).</summary>
        /// <param name="img">The multi-channel image.</param>
        /// <returns>The planes.</returns>
        /// <exception cref="OpenCVException">OpenCV rejects the input.</exception>
        public Mat[] split(Mat img)
        {
            using (var vec = new VectorOfMat())
            {
                NativeMethods.Call(() => cv2_native_api.core_split(img, vec));
                GC.KeepAlive(img);
                return vec.ToArray();
            }
        }

        /// <summary><c>cv2.hconcat(src)</c>: the images side by side (same height and type).</summary>
        /// <param name="imgs">The images.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">The heights or types differ.</exception>
        public Mat hconcat(params Mat[] imgs)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_hconcat1(imgs.Select(x => (IntPtr)x).ToArray(), (uint)imgs.Length, dst.OutputArray));
            GC.KeepAlive(imgs);
            return dst;
        }

        /// <summary><c>cv2.vconcat(src)</c>: the images stacked vertically (same width and type).</summary>
        /// <param name="imgs">The images.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="OpenCVException">The widths or types differ.</exception>
        public Mat vconcat(params Mat[] imgs)
        {
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.core_vconcat1(imgs.Select(x => (IntPtr)x).ToArray(), (uint)imgs.Length, dst.OutputArray));
            GC.KeepAlive(imgs);
            return dst;
        }
    }
}
