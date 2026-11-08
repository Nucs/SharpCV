using System;
using System.Collections.Generic;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     OpenCV's <c>cv::Scalar</c> (four doubles, laid out as the native <c>MyCvScalar</c>): a colour in the image's
    ///     channel order (BGR for colour images), or a per-channel value. cv2 accepts a number or a tuple of up to four;
    ///     so do the implicit conversions here, with the missing components 0.
    /// </summary>
    public struct Scalar
    {
        /// <summary>The first component (blue for a BGR colour).</summary>
        public double Val0 { get; }

        /// <summary>The second component (green).</summary>
        public double Val1 { get; }

        /// <summary>The third component (red).</summary>
        public double Val2 { get; }

        /// <summary>The fourth component (alpha), 0 unless given.</summary>
        public double Val3 { get; }

        /// <summary>A three-component scalar; the fourth is 0.</summary>
        /// <param name="v0">The first component.</param>
        /// <param name="v1">The second component.</param>
        /// <param name="v2">The third component.</param>
        public Scalar(double v0, double v1, double v2)
        {
            Val0 = v0;
            Val1 = v1;
            Val2 = v2;
            Val3 = 0;
        }

        /// <summary>A four-component scalar.</summary>
        /// <param name="v0">The first component.</param>
        /// <param name="v1">The second component.</param>
        /// <param name="v2">The third component.</param>
        /// <param name="v3">The fourth component.</param>
        public Scalar(double v0, double v1, double v2, double v3)
        {
            Val0 = v0;
            Val1 = v1;
            Val2 = v2;
            Val3 = v3;
        }

        /// <summary><c>cv::Scalar::all(v)</c>: every component <paramref name="v"/>.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The scalar.</returns>
        public static Scalar All(double v) => new Scalar(v, v, v, v);

        /// <summary>
        ///     <c>cv::morphologyDefaultBorderValue()</c> (every component <see cref="double.MaxValue"/>): the constant
        ///     border OpenCV's morphology treats as neutral (no erosion from the edge). OpenCV compares all four
        ///     components, so it is only recognized when passed exactly.
        /// </summary>
        public static Scalar MorphologyDefaultBorderValue => All(double.MaxValue);

        /// <summary>A colour from a (b, g, r) tuple.</summary>
        /// <param name="vals">The components.</param>
        /// <returns>The scalar.</returns>
        public static implicit operator Scalar((double, double, double) vals)
            => new Scalar(vals.Item1, vals.Item2, vals.Item3);

        /// <summary>A colour from a (b, g, r, a) tuple.</summary>
        /// <param name="vals">The components.</param>
        /// <returns>The scalar.</returns>
        public static implicit operator Scalar((double, double, double, double) vals)
            => new Scalar(vals.Item1, vals.Item2, vals.Item3, vals.Item4);

        /// <summary>A single value (a grayscale colour), as cv2 accepts a bare number; the other components are 0.</summary>
        /// <param name="v">The value.</param>
        /// <returns>The scalar.</returns>
        public static implicit operator Scalar(double v)
            => new Scalar(v, 0, 0, 0);
    }
}
