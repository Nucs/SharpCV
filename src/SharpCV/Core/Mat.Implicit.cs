using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace SharpCV
{
    public partial class Mat
    {
        /// <summary>The native <c>cv::Mat*</c>, for P/Invoke arguments; <see cref="IntPtr.Zero"/> for null or a disposed
        ///     Mat (OpenCV then sees no array).</summary>
        /// <param name="mat">The Mat.</param>
        /// <returns>Its handle.</returns>
        public static implicit operator IntPtr(Mat mat)
            => mat?._handle ?? IntPtr.Zero;

        /// <summary>Wraps a <c>cv::Mat*</c> and takes ownership of it (see <see cref="Mat(IntPtr)"/>).</summary>
        /// <param name="handle">The pointer.</param>
        /// <returns>The owning Mat.</returns>
        public static implicit operator Mat(IntPtr handle)
            => new Mat(handle);

        /// <summary>
        ///     The pixels as a zero-copy view (<see cref="data"/>), so a Mat can be used wherever an NDArray is expected:
        ///     <c>NDArray img = cv2.imread(path);</c>. The view outlives the Mat.
        /// </summary>
        /// <param name="mat">The Mat; null gives null.</param>
        /// <returns>The view, or null for a null or empty Mat (opencv-python's <c>None</c>).</returns>
        /// <exception cref="ObjectDisposedException">The Mat was disposed.</exception>
        public static implicit operator NDArray(Mat mat)
            => mat?.AsNDArray(MatCodecMode.View);

        /// <summary>
        ///     A Mat over the array (<see cref="Mat(NDArray, MatCodecMode)"/> in <see cref="MatCodecMode.Auto"/>): shared
        ///     when the layout fits, copied otherwise, as opencv-python does. The caller owns the Mat: dispose it, or it
        ///     waits for the finalizer with any copy it made.
        /// </summary>
        /// <param name="nd">The array; null gives null.</param>
        /// <returns>The Mat, or null.</returns>
        /// <exception cref="OpenCVException">The dtype has no OpenCV depth or the array has too many dimensions.</exception>
        /// <remarks>
        ///     <c>cv2.*</c> functions do not go through this conversion: their image parameters are
        ///     <see cref="InputArray"/> / <see cref="InputOutputArray"/>, which convert an NDArray for the one call and
        ///     release it when the call returns (and draw into an NDArray in place, never into a copy). It remains for
        ///     the places that keep a Mat (<c>Mat m = nd;</c>, a <c>Mat[]</c> of contours, <see cref="cv_api.imcrop"/>).
        /// </remarks>
        public static implicit operator Mat(NDArray nd)
            => nd is null ? null : new Mat(nd, MatCodecMode.Auto);
    }
}
