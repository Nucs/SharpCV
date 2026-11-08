using System;
using NumSharp;

namespace SharpCV
{
    /// <summary>
    ///     An image argument a cv2 function READS: a <see cref="SharpCV.Mat"/> or a NumSharp <see cref="NDArray"/>, both
    ///     converting implicitly, as OpenCV's own <c>cv::InputArray</c> proxy takes a Mat, a vector or a matrix expression.
    ///     <c>cv2.cvtColor(mat, ...)</c> and <c>cv2.cvtColor(ndarray, ...)</c> call the same function.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     An NDArray argument becomes a Mat by opencv-python's rules (shared when the layout fits cv::Mat, copied
    ///     otherwise; see <see cref="SharpCV.Mat(NDArray, MatCodecMode)"/>) for the duration of ONE call, and that Mat and
    ///     any copy are released when the call returns. Until SharpCV 0.20 the functions took a <see cref="SharpCV.Mat"/>
    ///     and the NDArray reached them through the implicit conversion, which left the temporary Mat (and, for a 1080p
    ///     <c>[..., ::-1]</c> view, its 6 MB copy) to the finalizer.
    ///     </para>
    ///     <para>
    ///     A <see cref="SharpCV.Mat"/> argument is used as it is and never disposed by the call. The struct only remembers
    ///     what it was given, so holding one in a variable and passing it twice is fine: each call converts again.
    ///     </para>
    /// </remarks>
    public readonly struct InputArray
    {
        /// <summary>The caller's Mat, or null.</summary>
        internal readonly Mat Mat;

        /// <summary>The caller's NDArray, or null.</summary>
        internal readonly NDArray Array;

        /// <summary>Remembers the argument.</summary>
        /// <param name="mat">A Mat, or null.</param>
        /// <param name="array">An NDArray, or null.</param>
        private InputArray(Mat mat, NDArray array)
        {
            Mat = mat;
            Array = array;
        }

        /// <summary>A Mat argument, used as it is.</summary>
        /// <param name="mat">The Mat.</param>
        /// <returns>The argument.</returns>
        public static implicit operator InputArray(Mat mat) => new InputArray(mat, null);

        /// <summary>An NDArray argument, converted for each call by opencv-python's rules and released after it.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The argument.</returns>
        public static implicit operator InputArray(NDArray array) => new InputArray(null, array);

        /// <summary>
        ///     The Mat this argument is passed to OpenCV as during one call; dispose the result after the call (it disposes
        ///     only a Mat it made).
        /// </summary>
        /// <param name="name">The parameter name, for refusal messages (cv2 names the argument: <c>src</c>,
        ///     <c>kernel</c>).</param>
        /// <param name="ndMat">True for an n-D blob parameter (cv2's <c>CV_ND</c>): a 3-D NDArray stays three
        ///     dimensions instead of becoming a multi-channel image.</param>
        /// <returns>The argument's Mat for this call.</returns>
        /// <exception cref="ArgumentNullException">The argument is null.</exception>
        /// <exception cref="OpenCVException">The NDArray's dtype has no OpenCV depth, or it has too many
        ///     dimensions.</exception>
        /// <remarks>A disposed Mat argument is not detected here: the wrapper's first read of its native handle throws
        ///     <see cref="ObjectDisposedException"/>.</remarks>
        internal MatArgument Open(string name, bool ndMat = false)
        {
            if (Mat is not null)
                return new MatArgument(Mat, temporary: false);
            // `is not null`: NDArray overloads != element-wise.
            if (Array is not null)
                return new MatArgument(new Mat(Array, MatCodecMode.Auto, name, output: false, ndMat), temporary: true);
            throw new ArgumentNullException(name);
        }
    }

    /// <summary>
    ///     An image argument a cv2 function draws INTO (rectangle, putText, drawContours): a <see cref="SharpCV.Mat"/> or a
    ///     NumSharp <see cref="NDArray"/>, OpenCV's <c>cv::InputOutputArray</c>. An NDArray is drawn into in place, so it
    ///     must be writeable and viewable as a Mat; cv2 refuses the same arrays ("Layout of the output array img is
    ///     incompatible with cv::Mat"), because drawing on a copy would be lost.
    /// </summary>
    /// <remarks>One parameter type serves a Mat target and an NDArray target; an NDArray target is never copied, so a
    ///     drawing can never silently land in a temporary.</remarks>
    public readonly struct InputOutputArray
    {
        /// <summary>The caller's Mat, or null.</summary>
        internal readonly Mat Mat;

        /// <summary>The caller's NDArray, or null.</summary>
        internal readonly NDArray Array;

        /// <summary>Remembers the argument.</summary>
        /// <param name="mat">A Mat, or null.</param>
        /// <param name="array">An NDArray, or null.</param>
        private InputOutputArray(Mat mat, NDArray array)
        {
            Mat = mat;
            Array = array;
        }

        /// <summary>A Mat argument, drawn into as it is.</summary>
        /// <param name="mat">The Mat.</param>
        /// <returns>The argument.</returns>
        public static implicit operator InputOutputArray(Mat mat) => new InputOutputArray(mat, null);

        /// <summary>An NDArray argument, drawn into in place through a Mat view made for each call.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The argument.</returns>
        public static implicit operator InputOutputArray(NDArray array) => new InputOutputArray(null, array);

        /// <summary>
        ///     The Mat OpenCV draws into during one call; dispose the result after the call (it disposes only a Mat it made).
        /// </summary>
        /// <param name="name">The parameter name, for refusal messages.</param>
        /// <returns>The argument's Mat for this call.</returns>
        /// <exception cref="ArgumentNullException">The argument is null.</exception>
        /// <exception cref="OpenCVException">The NDArray is read-only, its dtype has no depth, or its layout would need a
        ///     copy.</exception>
        internal MatArgument Open(string name)
        {
            if (Mat is not null)
                return new MatArgument(Mat, temporary: false);
            // `is not null`: NDArray overloads != element-wise.
            if (Array is not null)
                return new MatArgument(SharpCV.Mat.ForOutput(Array, name), temporary: true);
            throw new ArgumentNullException(name);
        }
    }

    /// <summary>
    ///     One argument's Mat for the duration of one call. <see cref="Dispose"/> releases a Mat made for the call (an
    ///     NDArray's view or copy) and keeps a caller's Mat reachable up to that point, so a wrapper needs no separate
    ///     <c>GC.KeepAlive</c> for it.
    /// </summary>
    internal readonly struct MatArgument : IDisposable
    {
        /// <summary>The Mat to hand to OpenCV.</summary>
        public readonly Mat Mat;

        /// <summary>True when <see cref="Mat"/> was made for this call and is disposed with it.</summary>
        private readonly bool _temporary;

        /// <summary>Pairs a Mat with its ownership.</summary>
        /// <param name="mat">The Mat.</param>
        /// <param name="temporary">True when the call owns it.</param>
        public MatArgument(Mat mat, bool temporary)
        {
            Mat = mat;
            _temporary = temporary;
        }

        /// <summary>Releases a Mat made for the call (its native header and any copy, at once); a caller's Mat is only
        ///     kept reachable until here.</summary>
        public void Dispose()
        {
            if (_temporary)
                Mat.Dispose();
            else
                GC.KeepAlive(Mat);
        }
    }
}
