using System;
using System.Collections.Generic;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     <c>cv2.moments</c>'s result: the spatial moments OpenCV computed, and the central and normalized central
    ///     moments derived from them with <c>cv::Moments</c>' own formulas (so the values match cv2's dictionary).
    /// </summary>
    public partial class Moments
    {
        /// <summary>
        /// spatial moments
        /// </summary>
        public double M00, M10, M01, M20, M11, M02, M30, M21, M12, M03;

        /// <summary>
        /// central moments
        /// </summary>
        public double Mu20, Mu11, Mu02, Mu30, Mu21, Mu12, Mu03;

        /// <summary>
        /// central normalized moments
        /// </summary>
        public double Nu20, Nu11, Nu02, Nu30, Nu21, Nu12, Nu03;

        /// <summary>Computes the moments of an image or contour (<c>cv2.moments(img, binaryImage)</c>).</summary>
        /// <param name="img">An image (single channel) or a contour: a Mat or an NDArray, read for the call.</param>
        /// <param name="binaryImage">True counts every non-zero pixel as 1.</param>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input (a multi-channel image), or cv2 refuses the
        ///     NDArray.</exception>
        public Moments(InputArray img, bool binaryImage = false)
        {
            using var array = img.Open("array");
            NativeStruct m = default;
            NativeMethods.Call(() => cv2_native_api.imgproc_moments(array.Mat.InputArray, binaryImage ? 1 : 0, out m));
            Initialize(m.m00, m.m10, m.m01, m.m20, m.m11, m.m02, m.m30, m.m21, m.m12, m.m03);
        }

        /// <summary>The pre-0.15 Mat constructor, kept for source compatibility; forwards to the InputArray one. C# 13+
        ///     prefers that constructor for Mat and NDArray arguments; C# 12 binds an NDArray argument here through the
        ///     deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="Moments(InputArray, bool)"/>
        [System.Runtime.CompilerServices.OverloadResolutionPriority(-1)]
        public Moments(Mat img, bool binaryImage = false)
            : this((InputArray)img, binaryImage)
        {
        }

        /// <summary>
        ///     Stores the spatial moments and derives the central and normalized ones exactly as the
        ///     <c>cv::Moments(m00, ..., m03)</c> constructor does.
        /// </summary>
        /// <param name="m00">The zeroth moment (the area / mass).</param>
        /// <param name="m10">First moment in x.</param>
        /// <param name="m01">First moment in y.</param>
        /// <param name="m20">Second moment in x.</param>
        /// <param name="m11">Mixed second moment.</param>
        /// <param name="m02">Second moment in y.</param>
        /// <param name="m30">Third moment in x.</param>
        /// <param name="m21">Mixed third moment (x^2 y).</param>
        /// <param name="m12">Mixed third moment (x y^2).</param>
        /// <param name="m03">Third moment in y.</param>
        private void Initialize(double m00, double m10, double m01, double m20, double m11,
                       double m02, double m30, double m21, double m12, double m03)
        {
            M00 = m00;
            M10 = m10;
            M01 = m01;
            M20 = m20;
            M11 = m11;
            M02 = m02;
            M30 = m30;
            M21 = m21;
            M12 = m12;
            M03 = m03;

            double cx = 0, cy = 0, invM00 = 0;
            if (Math.Abs(M00) > double.Epsilon)
            {
                invM00 = 1.0 / M00;
                cx = M10 * invM00;
                cy = M01 * invM00;
            }

            Mu20 = M20 - M10 * cx;
            Mu11 = M11 - M10 * cy;
            Mu02 = M02 - M01 * cy;

            Mu30 = M30 - cx * (3 * Mu20 + cx * M10);
            Mu21 = M21 - cx * (2 * Mu11 + cx * M01) - cy * Mu20;
            Mu12 = M12 - cy * (2 * Mu11 + cy * M10) - cx * Mu02;
            Mu03 = M03 - cy * (3 * Mu02 + cy * M01);

            var invSqrtM00 = Math.Sqrt(Math.Abs(invM00));
            var s2 = invM00 * invM00;
            var s3 = s2 * invSqrtM00;

            Nu20 = Mu20 * s2;
            Nu11 = Mu11 * s2;
            Nu02 = Mu02 * s2;
            Nu30 = Mu30 * s3;
            Nu21 = Mu21 * s3;
            Nu12 = Mu12 * s3;
            Nu03 = Mu03 * s3;
        }
    }
}
