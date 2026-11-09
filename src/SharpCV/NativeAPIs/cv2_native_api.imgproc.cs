using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace SharpCV
{
    internal partial class cv2_native_api
    {
        /// <summary><c>cv::cvtColor(src, dst, code, dstCn)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="code">The <see cref="ColorConversionCodes"/> code.</param>
        /// <param name="dstCnt">The destination channel count; 0 derives it from the code.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_cvtColor(IntPtr src, IntPtr dst, int code, int dstCnt);

        /// <summary><c>cv::contourArea(contour, oriented)</c>.</summary>
        /// <param name="contour">The contour (an <c>_InputArray</c> of points).</param>
        /// <param name="oriented">1 returns a signed area that encodes the orientation.</param>
        /// <param name="output">Receives the area.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_contourArea_InputArray(IntPtr contour, int oriented, out double output);

        /// <summary><c>cv::threshold(src, dst, thresh, maxval, type)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="thresh">The threshold (ignored with OTSU / TRIANGLE).</param>
        /// <param name="maxval">The value written for pixels that pass.</param>
        /// <param name="type">The <see cref="ThresholdTypes"/> flags.</param>
        /// <param name="output">Receives the threshold actually used.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_threshold(IntPtr src, IntPtr dst, double thresh, double maxval, int type, out double output);

        /// <summary><c>cv::resize(src, dst, dsize, fx, fy, interpolation)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="dsize">The output size (width, height); (0, 0) derives it from the factors.</param>
        /// <param name="fx">The horizontal scale factor when <paramref name="dsize"/> is (0, 0).</param>
        /// <param name="fy">The vertical scale factor when <paramref name="dsize"/> is (0, 0).</param>
        /// <param name="interpolation">The <see cref="InterpolationFlags"/> method.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_resize(IntPtr src, IntPtr dst, Size dsize, double fx, double fy, int interpolation);

        /// <summary><c>new cv::Mat(cv::getRotationMatrix2D(center, angle, scale))</c>: a 2 x 3 float64 affine matrix.
        ///     The export takes a FLOAT point; SharpCV passed an int <see cref="Point"/> until 0.15, so the centre's int
        ///     bits were read as floats near 5e-43 and every rotation turned about the origin.</summary>
        /// <param name="center">The rotation centre.</param>
        /// <param name="angle">The angle in degrees, counter-clockwise.</param>
        /// <param name="scale">The isotropic scale.</param>
        /// <param name="output">Receives the new Mat.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_getRotationMatrix2D(Point2f center, double angle, double scale, out IntPtr output);

        /// <summary><c>cv::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="m">The 2 x 3 matrix (an <c>_InputArray</c>).</param>
        /// <param name="dsize">The output size (width, height).</param>
        /// <param name="flags">The <see cref="InterpolationFlags"/>, optionally with <c>WARP_INVERSE_MAP</c>.</param>
        /// <param name="borderMode">The <see cref="BorderTypes"/> mode.</param>
        /// <param name="borderValue">The value of a constant border.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_warpAffine(IntPtr src, IntPtr dst, IntPtr m, Size dsize, int flags, int borderMode, Scalar borderValue);

        /// <summary><c>cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>, 8-bit single channel).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="maxValue">The value written for pixels that pass.</param>
        /// <param name="method">The <see cref="AdaptiveThresholdTypes"/> method.</param>
        /// <param name="type">The <see cref="ThresholdTypes"/> type (BINARY or BINARY_INV).</param>
        /// <param name="blockSize">The neighbourhood size (odd).</param>
        /// <param name="delta">The constant subtracted from the neighbourhood mean.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_adaptiveThreshold(IntPtr src, IntPtr dst, double maxValue, int method, int type, int blockSize, double delta);

        /// <summary><c>cv::pyrUp(src, dst, dstsize, borderType)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="dstsize">The output size; (0, 0) doubles the input.</param>
        /// <param name="borderType">The <see cref="BorderTypes"/> mode.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_pyrUp(IntPtr src, IntPtr dst, Size dstsize, int borderType);

        /// <summary><c>cv::pyrDown(src, dst, dstsize, borderType)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="dstsize">The output size; (0, 0) halves the input (rounding up).</param>
        /// <param name="borderType">The <see cref="BorderTypes"/> mode.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_pyrDown(IntPtr src, IntPtr dst, Size dstsize, int borderType);

        /// <summary><c>cv::calcBackProject</c> over an array of Mats (declared, not wrapped yet).</summary>
        /// <param name="images">The source images (cv::Mat pointers).</param>
        /// <param name="nimages">Their count.</param>
        /// <param name="channels">The channel indices used.</param>
        /// <param name="hist">The histogram (an <c>_InputArray</c>).</param>
        /// <param name="backProject">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="ranges">The bin boundaries per dimension.</param>
        /// <param name="uniform">1 for uniform bins.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_calcBackProject(IntPtr[] images, int nimages, int[] channels, IntPtr hist, IntPtr backProject, IntPtr[] ranges, int uniform);

        /// <summary><c>cv::rectangle(img, pt1, pt2, color, thickness, lineType, shift)</c>, drawn in place.</summary>
        /// <param name="img">The image (an <c>_OutputArray</c> standing for the <c>_InputOutputArray</c>).</param>
        /// <param name="pt1">One corner.</param>
        /// <param name="pt2">The opposite corner.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The <see cref="LineTypes"/> type.</param>
        /// <param name="shift">The number of fractional bits in the coordinates.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_rectangle_InputOutputArray_Point(IntPtr img, Point pt1, Point pt2, Scalar color, int thickness, LineTypes lineType, int shift);

        /// <summary><c>cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)</c>,
        ///     drawn in place.</summary>
        /// <param name="img">The image (an <c>_OutputArray</c> standing for the <c>_InputOutputArray</c>).</param>
        /// <param name="text">The text (ANSI; Hershey fonts draw ASCII only).</param>
        /// <param name="org">The bottom-left corner of the text.</param>
        /// <param name="fontFace">The <see cref="HersheyFonts"/> font.</param>
        /// <param name="fontScale">The font scale.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The stroke thickness.</param>
        /// <param name="lineType">The <see cref="LineTypes"/> type.</param>
        /// <param name="bottomLeftOrigin">1 when the image origin is the bottom-left corner.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern ExceptionStatus imgproc_putText(IntPtr img, string text, Point org,
            int fontFace, double fontScale, Scalar color,
            int thickness, int lineType, int bottomLeftOrigin);

        /// <summary><c>cv::getTextSize(text, fontFace, fontScale, thickness, &amp;baseLine)</c>.</summary>
        /// <param name="text">The text.</param>
        /// <param name="fontFace">The <see cref="HersheyFonts"/> font.</param>
        /// <param name="fontScale">The font scale.</param>
        /// <param name="thickness">The stroke thickness.</param>
        /// <param name="baseLine">Receives the baseline's y offset.</param>
        /// <param name="returnValue">Receives the text box size.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        internal static extern ExceptionStatus imgproc_getTextSize(string text, int fontFace,
            double fontScale, int thickness, out int baseLine, out Size returnValue);

        /// <summary><c>new cv::Mat(cv::getStructuringElement(shape, ksize, anchor))</c>.</summary>
        /// <param name="shape">The <see cref="MorphShapes"/> shape.</param>
        /// <param name="ksize">The kernel size (width, height).</param>
        /// <param name="anchor">The anchor; (-1, -1) is the centre and matters for MORPH_CROSS only.</param>
        /// <param name="output">Receives the 8-bit kernel.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_getStructuringElement(int shape, Size ksize, Point anchor, out IntPtr output);

        /// <summary><c>cv::morphologyEx(src, dst, op, kernel, anchor, iterations, borderType, borderValue)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="op">The <see cref="MorphTypes"/> operation.</param>
        /// <param name="kernel">The structuring element (an <c>_InputArray</c>).</param>
        /// <param name="anchor">The anchor; (-1, -1) is the kernel centre.</param>
        /// <param name="iterations">How many times erosion and dilation are applied.</param>
        /// <param name="borderType">The <see cref="BorderTypes"/> mode.</param>
        /// <param name="borderValue">The constant border's value; all four components <c>DBL_MAX</c>
        ///     (<c>morphologyDefaultBorderValue()</c>) make the border neutral for the operation.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_morphologyEx(IntPtr src, IntPtr dst, int op, IntPtr kernel, Point anchor,
            int iterations, int borderType, Scalar borderValue);

        /// <summary><c>cv::findContours(image, *contours, *hierarchy, mode, method, offset)</c> into CALLER-allocated
        ///     vectors. Through OpenCvSharp 4.4 this export allocated the vectors and returned them through out
        ///     parameters; 4.13 dereferences the pointers it is given, so passing the address of a local (SharpCV's old
        ///     <c>out IntPtr</c>) makes OpenCV construct a vector on the stack.</summary>
        /// <param name="image">The 8-bit binary image (an <c>_InputArray</c>; OpenCV no longer modifies it).</param>
        /// <param name="contours">A <c>std::vector&lt;std::vector&lt;cv::Point&gt;&gt;</c> from
        ///     <see cref="vector_vector_Point_new1"/>.</param>
        /// <param name="hierarchy">A <c>std::vector&lt;cv::Vec4i&gt;</c> from <see cref="vector_Vec4i_new1"/>.</param>
        /// <param name="mode">The <see cref="RetrievalModes"/> mode.</param>
        /// <param name="method">The <see cref="ContourApproximationModes"/> method.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_findContours1_vector(IntPtr image, IntPtr contours,
            IntPtr hierarchy, int mode, int method, Point offset);

        /// <summary><c>cv::findContours(image, *contours, hierarchy, mode, method, offset)</c> with every contour as a
        ///     Mat (N x 1, CV_32SC2) in a CALLER-allocated vector (see <see cref="imgproc_findContours1_vector"/>).</summary>
        /// <param name="image">The 8-bit binary image (an <c>_InputArray</c>).</param>
        /// <param name="contours">A <c>std::vector&lt;cv::Mat&gt;</c> from <see cref="vector_Mat_new1"/>.</param>
        /// <param name="hierarchy">The hierarchy destination (an <c>_OutputArray</c>, 1 x N CV_32SC4).</param>
        /// <param name="mode">The <see cref="RetrievalModes"/> mode.</param>
        /// <param name="method">The <see cref="ContourApproximationModes"/> method.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_findContours1_OutputArray(IntPtr image, IntPtr contours,
            IntPtr hierarchy, int mode, int method, Point offset);

        /// <summary><c>cv::drawContours</c> over contours given as pinned point arrays, drawn in place.</summary>
        /// <param name="image">The image (an <c>_OutputArray</c> standing for the <c>_InputOutputArray</c>).</param>
        /// <param name="contours">One pointer per contour to its first <see cref="Point"/>.</param>
        /// <param name="contoursSize1">The number of contours.</param>
        /// <param name="contoursSize2">The point count of each contour.</param>
        /// <param name="contourIdx">The contour to draw; negative draws all.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The <see cref="LineTypes"/> code.</param>
        /// <param name="hierarchy">The hierarchy as <c>cv::Vec4i</c>s, or null.</param>
        /// <param name="hiearchyLength">The number of hierarchy entries.</param>
        /// <param name="maxLevel">The deepest nested contour level drawn.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_drawContours_vector(IntPtr image,
            IntPtr[] contours, int contoursSize1, int[] contoursSize2,
            int contourIdx, Scalar color, int thickness, int lineType,
            IntPtr hierarchy, int hiearchyLength, int maxLevel, Point offset);

        /// <summary><c>cv::drawContours</c> over contours given as Mats, drawn in place.</summary>
        /// <param name="image">The image (an <c>_OutputArray</c> standing for the <c>_InputOutputArray</c>).</param>
        /// <param name="contours">The contours (cv::Mat pointers, N x 1 CV_32SC2 each).</param>
        /// <param name="contoursLength">Their count.</param>
        /// <param name="contourIdx">The contour to draw; negative draws all.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The <see cref="LineTypes"/> type.</param>
        /// <param name="hierarchy">The hierarchy as an <c>_InputArray</c>, or null for none. SharpCV passed the Mat
        ///     itself here until 0.15, which OpenCV then read as an <c>_InputArray</c>.</param>
        /// <param name="maxLevel">The deepest nested contour level drawn.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_drawContours_InputArray(IntPtr image,
            IntPtr[] contours, int contoursLength,
            int contourIdx, Scalar color, int thickness, LineTypes lineType,
            IntPtr hierarchy, int maxLevel, Point offset);

        /// <summary><c>cv::approxPolyDP(curve, approxCurve, epsilon, closed)</c>.</summary>
        /// <param name="curve">The curve (an <c>_InputArray</c> of points).</param>
        /// <param name="approxCurve">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="epsilon">The maximal distance between the curve and its approximation.</param>
        /// <param name="closed">True for a closed curve; marshalled as the export's <c>int</c>.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_approxPolyDP_InputArray(IntPtr curve, IntPtr approxCurve,
            double epsilon, [MarshalAs(UnmanagedType.Bool)] bool closed);

        /// <summary><c>cv::minAreaRect(points)</c>.</summary>
        /// <param name="points">The points (an <c>_InputArray</c>).</param>
        /// <param name="output">Receives the rectangle (centre, size, angle: five floats).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_minAreaRect_InputArray(IntPtr points, out RotatedRect output);

        /// <summary><c>cv::moments(arr, binaryImage)</c>: the spatial and central moments (normalized ones are
        ///     derived by <see cref="Moments"/>).</summary>
        /// <param name="arr">The image or contour (an <c>_InputArray</c>).</param>
        /// <param name="binaryImage">1 treats every non-zero pixel as 1.</param>
        /// <param name="output">Receives the moments.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_moments(IntPtr arr, int binaryImage, out Moments.NativeStruct output);

        /// <summary><c>cv::medianBlur(src, dst, ksize)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="kSize">The aperture (odd, greater than 1).</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_medianBlur(IntPtr src, IntPtr dst, int kSize);

        /// <summary><c>cv::filter2D(src, dst, ddepth, kernel, anchor, delta, borderType)</c>.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="ddepth">The destination depth; -1 (<see cref="MatType.Unknown"/>) keeps the source's.</param>
        /// <param name="kernel">The correlation kernel (an <c>_InputArray</c>, single channel float).</param>
        /// <param name="anchor">The anchor; (-1, -1) is the kernel centre.</param>
        /// <param name="delta">Added to every result.</param>
        /// <param name="borderType">The <see cref="BorderTypes"/> mode.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_filter2D(IntPtr src, IntPtr dst, MatType ddepth, IntPtr kernel, Point anchor,
            double delta, BorderTypes borderType);

        /// <summary><c>cv::blur(src, dst, ksize, anchor, borderType)</c>: the normalized box filter.</summary>
        /// <param name="src">The source (an <c>_InputArray</c>).</param>
        /// <param name="dst">The destination (an <c>_OutputArray</c>).</param>
        /// <param name="kSize">The box size (width, height).</param>
        /// <param name="anchor">The anchor; (-1, -1) is the box centre.</param>
        /// <param name="borderType">The <see cref="BorderTypes"/> mode.</param>
        /// <returns>The call status.</returns>
        [DllImport(OpenCvDllName, CallingConvention = CallingConvention.Cdecl)]
        internal static extern ExceptionStatus imgproc_blur(IntPtr src, IntPtr dst, Size kSize, Point anchor, BorderTypes borderType);
    }
}
