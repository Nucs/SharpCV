using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;

namespace SharpCV
{
    public partial class Mat
    {
        /// <summary>The native <c>cv::Mat*</c>, for P/Invoke arguments; <see cref="IntPtr.Zero"/> for null or a disposed
        ///     Mat (OpenCV then sees no array). A deferred Mat (an implicit NDArray conversion) builds its header
        ///     here.</summary>
        /// <param name="mat">The Mat.</param>
        /// <returns>Its handle.</returns>
        /// <exception cref="OpenCVException">A deferred Mat's header could not be built.</exception>
        public static implicit operator IntPtr(Mat mat)
            => mat is null ? IntPtr.Zero : mat._handle != IntPtr.Zero ? mat._handle : mat.IsPending ? mat.Handle : IntPtr.Zero;

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
        ///     A Mat over the array, as <see cref="Mat(NDArray, MatCodecMode)"/> in <see cref="MatCodecMode.Auto"/> builds
        ///     it (shared when the layout fits, copied otherwise, as opencv-python does), but DEFERRED: it holds a counted
        ///     view of the array and builds its native header and any copy only when something first needs them. The
        ///     caller owns the Mat: dispose it.
        /// </summary>
        /// <param name="nd">The array; null gives null.</param>
        /// <returns>The Mat, or null.</returns>
        /// <exception cref="OpenCVException">The dtype has no OpenCV depth or the array has too many dimensions (raised
        ///     here, at conversion).</exception>
        /// <remarks>
        ///     <para>
        ///     Why deferred: a C# 12 compiler (net8.0's default) ignores the priority of the pre-0.15 overloads and binds
        ///     an NDArray argument to <c>cvtColor(Mat ...)</c> through this conversion. Built eagerly, that temporary Mat
        ///     (and the copy a reversed view needs) waited for the finalizer: measured on 300 1080p
        ///     <c>cvtColor(nd[..., ::-1])</c> calls, +53 MB and 10 gen2 collections where the proxy path has none. A
        ///     deferred Mat that reaches a cv2 function unbuilt is read as its array instead, converted for that call
        ///     and released when it returns, so the caller's compiler no longer decides what an NDArray argument
        ///     costs.
        ///     </para>
        ///     <para>
        ///     Everything else behaves as the eager Mat did: <c>Mat m = nd;</c> keeps working with every member, a refusal
        ///     is raised at once, and the caller may dispose <paramref name="nd"/> right away.
        ///     </para>
        /// </remarks>
        public static implicit operator Mat(NDArray nd)
            => nd is null ? null : Deferred(nd);
    }
}
