using System;
using System.Collections.Generic;
using System.Text;

namespace SharpCV
{
    /// <summary>
    ///     OpenCV's <c>cv::Point2f</c> (two floats, laid out as the native <c>MyCvPoint2D32f</c>): a sub-pixel position,
    ///     e.g. a rotation centre. Integer, float and double tuples convert implicitly, as cv2 accepts any numbers.
    /// </summary>
    public struct Point2f
    {
        /// <summary>The x coordinate (column).</summary>
        public float X { get; }

        /// <summary>The y coordinate (row).</summary>
        public float Y { get; }

        /// <summary>A point from integer coordinates.</summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        public Point2f(int x, int y)
        {
            X = Convert.ToSingle(x);
            Y = Convert.ToSingle(y);
        }

        /// <summary>A point from float coordinates.</summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        public Point2f(float x, float y)
        {
            X = x;
            Y = y;
        }

        /// <summary>A point from a float tuple.</summary>
        /// <param name="vals">(x, y).</param>
        /// <returns>The point.</returns>
        public static implicit operator Point2f((float, float) vals)
            => new Point2f(vals.Item1, vals.Item2);

        /// <summary>A point from an integer tuple.</summary>
        /// <param name="vals">(x, y).</param>
        /// <returns>The point.</returns>
        public static implicit operator Point2f((int, int) vals)
            => new Point2f(vals.Item1, vals.Item2);

        /// <summary>A point from a double tuple, rounded to float as cv2 converts a Python float for a Point2f.</summary>
        /// <param name="vals">(x, y).</param>
        /// <returns>The point.</returns>
        public static implicit operator Point2f((double, double) vals)
            => new Point2f((float)vals.Item1, (float)vals.Item2);

        /// <summary>The point as <c>{X}x{Y}</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString()
        {
            return $"{X}x{Y}";
        }
    }
}
