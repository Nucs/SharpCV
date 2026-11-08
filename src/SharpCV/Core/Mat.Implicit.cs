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
        ///     A Mat over the array (<see cref="Mat(NDArray, MatCodecMode)"/> in <see cref="MatCodecMode.Auto"/>), so every
        ///     <c>cv2.*</c> function accepts an NDArray as an input: shared when the layout fits, copied otherwise, as
        ///     opencv-python does.
        /// </summary>
        /// <param name="nd">The array; null gives null.</param>
        /// <returns>The Mat, or null.</returns>
        /// <exception cref="OpenCVException">The dtype has no OpenCV depth or the array has too many dimensions.</exception>
        /// <remarks>
        ///     Functions that write INTO their image (rectangle, putText, drawContours) have NDArray overloads instead,
        ///     because an implicit copy would silently swallow the drawing; those demand a writeable view, as cv2 does.
        /// </remarks>
        public static implicit operator Mat(NDArray nd)
            => nd is null ? null : new Mat(nd, MatCodecMode.Auto);
    }
}
