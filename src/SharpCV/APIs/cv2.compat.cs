using System;
using System.Runtime.CompilerServices;
using NumSharp;

namespace SharpCV
{
    // The pre-0.15 overloads, whose image parameters were Mat (and NDArray for drawing); cv_api's remarks (cv2.core.cs)
    // explain the priority rule. Each forwards to its InputArray / InputOutputArray twin, so it computes the same result.
    public partial class cv_api
    {
        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="flip(InputArray, FLIP_MODE)"/>
        [OverloadResolutionPriority(-1)]
        public Mat flip(Mat img, FLIP_MODE mode)
            => flip((InputArray)img, mode);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="rotate(InputArray, RotateFlags)"/>
        [OverloadResolutionPriority(-1)]
        public Mat rotate(Mat img, RotateFlags flags)
            => rotate((InputArray)img, flags);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="split(InputArray)"/>
        [OverloadResolutionPriority(-1)]
        public Mat[] split(Mat img)
            => split((InputArray)img);

        /// <summary>The pre-0.15 overload for Mats, also the one a <c>Mat[]</c> variable binds to (it converts to neither
        ///     <c>InputArray[]</c> nor <c>InputArray</c>). C# 13+ binds separate Mat or NDArray arguments to the
        ///     <c>params InputArray[]</c> overload; C# 12 binds NDArray arguments here as deferred Mats, which go through
        ///     that same proxy path (<see cref="HasPending"/>), released at return.</summary>
        /// <inheritdoc cref="Concatenate(Mat[], bool)"/>
        /// <param name="imgs">The images.</param>
        /// <returns>A new image, the images side by side (same height and type).</returns>
        [OverloadResolutionPriority(-1)]
        public Mat hconcat(params Mat[] imgs)
            => HasPending(imgs) ? Concatenate(Array.ConvertAll(imgs, m => (InputArray)m), horizontal: true) : Concatenate(imgs, horizontal: true);

        /// <summary>The pre-0.15 overload for Mats, also the one a <c>Mat[]</c> variable binds to. C# 13+ binds separate
        ///     Mat or NDArray arguments to the <c>params InputArray[]</c> overload; C# 12 binds NDArray arguments here as
        ///     deferred Mats, which go through that same proxy path (<see cref="HasPending"/>), released at return.</summary>
        /// <inheritdoc cref="Concatenate(Mat[], bool)"/>
        /// <param name="imgs">The images.</param>
        /// <returns>A new image, the images stacked vertically (same width and type).</returns>
        [OverloadResolutionPriority(-1)]
        public Mat vconcat(params Mat[] imgs)
            => HasPending(imgs) ? Concatenate(Array.ConvertAll(imgs, m => (InputArray)m), horizontal: false) : Concatenate(imgs, horizontal: false);

        /// <summary>
        ///     Whether any image is a deferred Mat nothing has built yet: a C# 12 <c>hconcat(nd1, nd2)</c> binds the
        ///     <c>params Mat[]</c> overload through the implicit NDArray conversion, and routing such images through the
        ///     proxy path converts their arrays for the call instead of building Mats (and copies) left to the finalizer.
        /// </summary>
        /// <param name="imgs">The images (may be null; null elements count as not pending).</param>
        /// <returns>True when the proxy path should read them.</returns>
        private static bool HasPending(Mat[] imgs)
            => imgs is not null && Array.Exists(imgs, m => m is not null && m.IsPending);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="imshow(string, InputArray)"/>
        [OverloadResolutionPriority(-1)]
        public void imshow(string windownName, Mat img)
            => imshow(windownName, (InputArray)img);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="imwrite(string, InputArray, int[])"/>
        [OverloadResolutionPriority(-1)]
        public bool imwrite(string filename, Mat img, params int[] @params)
            => imwrite(filename, (InputArray)img, @params);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="imdecode(InputArray, IMREAD_COLOR)"/>
        [OverloadResolutionPriority(-1)]
        public Mat imdecode(Mat buf, IMREAD_COLOR flags)
            => imdecode((InputArray)buf, flags);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="imencode(string, InputArray, int[])"/>
        [OverloadResolutionPriority(-1)]
        public (bool, NDArray) imencode(string ext, Mat img, params int[] @params)
            => imencode(ext, (InputArray)img, @params);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="ImEncode(string, InputArray, out byte[], int[])"/>
        [OverloadResolutionPriority(-1)]
        public static bool ImEncode(string ext, Mat img, out byte[] buf, int[]? prms = null)
            => ImEncode(ext, (InputArray)img, out buf, prms);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="cvtColor(InputArray, ColorConversionCodes, int)"/>
        [OverloadResolutionPriority(-1)]
        public Mat cvtColor(Mat img, ColorConversionCodes code, int dstCnt = 0)
            => cvtColor((InputArray)img, code, dstCnt);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="contourArea(InputArray, bool)"/>
        [OverloadResolutionPriority(-1)]
        public double contourArea(Mat img, bool oriented = false)
            => contourArea((InputArray)img, oriented);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="adaptiveThreshold(InputArray, double, AdaptiveThresholdTypes, ThresholdTypes, int, double)"/>
        [OverloadResolutionPriority(-1)]
        public Mat adaptiveThreshold(Mat img, double maxValue, AdaptiveThresholdTypes method, ThresholdTypes type, int blockSize, double delta)
            => adaptiveThreshold((InputArray)img, maxValue, method, type, blockSize, delta);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="threshold(InputArray, double, double, ThresholdTypes)"/>
        [OverloadResolutionPriority(-1)]
        public (double, Mat) threshold(Mat img, double thresh, double maxval, ThresholdTypes type)
            => threshold((InputArray)img, thresh, maxval, type);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="pyrUp(InputArray)"/>
        [OverloadResolutionPriority(-1)]
        public Mat pyrUp(Mat img)
            => pyrUp((InputArray)img);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="pyrUp(InputArray, ValueTuple{int, int}, BorderTypes)"/>
        [OverloadResolutionPriority(-1)]
        public Mat pyrUp(Mat img, (int, int) dstSize, BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
            => pyrUp((InputArray)img, dstSize, borderType);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="pyrDown(InputArray, ValueTuple{int, int}, BorderTypes)"/>
        [OverloadResolutionPriority(-1)]
        public Mat pyrDown(Mat img, (int, int) dstSize, BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
            => pyrDown((InputArray)img, dstSize, borderType);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="pyrDown(InputArray)"/>
        [OverloadResolutionPriority(-1)]
        public Mat pyrDown(Mat img)
            => pyrDown((InputArray)img);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="resize(InputArray, ValueTuple{int, int}, double, double, InterpolationFlags)"/>
        [OverloadResolutionPriority(-1)]
        public Mat resize(Mat img, (int, int) dsize, double fx = 0, double fy = 0, InterpolationFlags interpolation = InterpolationFlags.INTER_LINEAR)
            => resize((InputArray)img, dsize, fx, fy, interpolation);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="rotate(InputArray, Point2f, double, double, InterpolationFlags, BorderTypes)"/>
        [OverloadResolutionPriority(-1)]
        public Mat rotate(Mat img, Point2f center, double angle, double scale, InterpolationFlags flags = InterpolationFlags.INTER_LINEAR, BorderTypes borderMode = BorderTypes.BORDER_CONSTANT)
            => rotate((InputArray)img, center, angle, scale, flags, borderMode);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds NDArray arguments here through
        ///     the deferred implicit conversion, and the call reads those arrays, released at return.</summary>
        /// <inheritdoc cref="warpAffine(InputArray, InputArray, Size, InterpolationFlags, BorderTypes, Nullable{Scalar})"/>
        [OverloadResolutionPriority(-1)]
        public Mat warpAffine(Mat src, Mat M, Size dsize, InterpolationFlags flags = InterpolationFlags.INTER_LINEAR, BorderTypes borderMode = BorderTypes.BORDER_CONSTANT, Scalar? borderValue = null)
            => warpAffine((InputArray)src, (InputArray)M, dsize, flags, borderMode, borderValue);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputOutputArray overload,
        ///     so it draws exactly the same pixels. C# 13+ prefers that overload for a Mat argument.</summary>
        /// <inheritdoc cref="rectangle(InputOutputArray, Point, Point, Scalar, int, LineTypes, int)"/>
        [OverloadResolutionPriority(-1)]
        public void rectangle(Mat img, Point leftTop, Point rightBottom, Scalar color, int thickness = 1, LineTypes type = LineTypes.LINE_8, int shift = 0)
            => rectangle((InputOutputArray)img, leftTop, rightBottom, color, thickness, type, shift);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputOutputArray
        ///     overload, so it draws into the array in place the same way (a C# 12 caller lands here, never on the Mat
        ///     overload, whose implicit copy would swallow the drawing).</summary>
        /// <inheritdoc cref="rectangle(InputOutputArray, Point, Point, Scalar, int, LineTypes, int)"/>
        [OverloadResolutionPriority(-1)]
        public void rectangle(NDArray img, Point leftTop, Point rightBottom, Scalar color, int thickness = 1, LineTypes type = LineTypes.LINE_8, int shift = 0)
            => rectangle((InputOutputArray)img, leftTop, rightBottom, color, thickness, type, shift);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputOutputArray overload,
        ///     so it draws exactly the same pixels. C# 13+ prefers that overload for a Mat argument.</summary>
        /// <inheritdoc cref="putText(InputOutputArray, string, Point, HersheyFonts, double, Scalar, int, LineTypes, bool)"/>
        [OverloadResolutionPriority(-1)]
        public void putText(Mat img, string text, Point org, HersheyFonts fontFace, double fontScale, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, bool bottomLeftOrigin = false)
            => putText((InputOutputArray)img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputOutputArray
        ///     overload, so it draws into the array in place the same way.</summary>
        /// <inheritdoc cref="putText(InputOutputArray, string, Point, HersheyFonts, double, Scalar, int, LineTypes, bool)"/>
        [OverloadResolutionPriority(-1)]
        public void putText(NDArray img, string text, Point org, HersheyFonts fontFace, double fontScale, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, bool bottomLeftOrigin = false)
            => putText((InputOutputArray)img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds NDArray arguments here through
        ///     the deferred implicit conversion, and the call reads those arrays, released at return.</summary>
        /// <inheritdoc cref="morphologyEx(InputArray, MorphTypes, InputArray, Nullable{Point}, int, BorderTypes, Nullable{Scalar})"/>
        [OverloadResolutionPriority(-1)]
        public Mat morphologyEx(Mat src, MorphTypes type, Mat kernel, Point? anchor = null, int iterations = 1, BorderTypes borderType = BorderTypes.BORDER_CONSTANT, Scalar? borderValue = null)
            => morphologyEx((InputArray)src, type, (InputArray)kernel, anchor, iterations, borderType, borderValue);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="findContoursAsPoints(InputArray, RetrievalModes, ContourApproximationModes, Point)"/>
        [OverloadResolutionPriority(-1)]
        public Point[][] findContoursAsPoints(Mat src, RetrievalModes mode, ContourApproximationModes method, Point offset = default)
            => findContoursAsPoints((InputArray)src, mode, method, offset);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="findContours(InputArray, RetrievalModes, ContourApproximationModes, Point)"/>
        [OverloadResolutionPriority(-1)]
        public (Mat[], Mat) findContours(Mat src, RetrievalModes mode, ContourApproximationModes method, Point offset = default)
            => findContours((InputArray)src, mode, method, offset);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputOutputArray overload,
        ///     so it draws exactly the same pixels. C# 13+ prefers that overload for a Mat argument.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, Point[][], int, Scalar, int, LineTypes, int, Point)"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(Mat image, Point[][] contours, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, int maxLevel = int.MaxValue, Point offset = default)
            => drawContours((InputOutputArray)image, contours, contourIdx, color, thickness, lineType, maxLevel, offset);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputOutputArray
        ///     overload, so it draws into the array in place the same way.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, Point[][], int, Scalar, int, LineTypes, int, Point)"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(NDArray image, Point[][] contours, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, int maxLevel = int.MaxValue, Point offset = default)
            => drawContours((InputOutputArray)image, contours, contourIdx, color, thickness, lineType, maxLevel, offset);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputOutputArray /
        ///     InputArray overload, so it draws exactly the same pixels. C# 13+ prefers that overload.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, InputArray, int, Scalar, int, LineTypes, Mat, int, Nullable{Point})"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(Mat image, Mat contour, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, Mat? hierarchy = null, int maxLevel = int.MaxValue, Point? offset = null)
            => drawContours((InputOutputArray)image, (InputArray)contour, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputOutputArray /
        ///     InputArray overload, so it draws into the array in place the same way.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, InputArray, int, Scalar, int, LineTypes, Mat, int, Nullable{Point})"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(NDArray image, Mat contour, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, Mat? hierarchy = null, int maxLevel = int.MaxValue, Point? offset = null)
            => drawContours((InputOutputArray)image, (InputArray)contour, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputOutputArray overload,
        ///     so it draws exactly the same pixels. C# 13+ prefers that overload for a Mat argument.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, Mat[], int, Scalar, int, LineTypes, Mat, int, Nullable{Point})"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(Mat image, Mat[] contours, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, Mat? hierarchy = null, int maxLevel = int.MaxValue, Point? offset = null)
            => drawContours((InputOutputArray)image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);

        /// <summary>The pre-0.15 NDArray overload, kept for source compatibility; forwards to the InputOutputArray
        ///     overload, so it draws into the array in place the same way.</summary>
        /// <inheritdoc cref="drawContours(InputOutputArray, Mat[], int, Scalar, int, LineTypes, Mat, int, Nullable{Point})"/>
        [OverloadResolutionPriority(-1)]
        public void drawContours(NDArray image, Mat[] contours, int contourIdx, Scalar color, int thickness = 1, LineTypes lineType = LineTypes.LINE_8, Mat? hierarchy = null, int maxLevel = int.MaxValue, Point? offset = null)
            => drawContours((InputOutputArray)image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="approxPolyDP(InputArray, double, bool)"/>
        [OverloadResolutionPriority(-1)]
        public Mat approxPolyDP(Mat curve, double epsilon, bool closed)
            => approxPolyDP((InputArray)curve, epsilon, closed);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="minAreaRect(InputArray)"/>
        [OverloadResolutionPriority(-1)]
        public RotatedRect minAreaRect(Mat points)
            => minAreaRect((InputArray)points);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="moments(InputArray, bool)"/>
        [OverloadResolutionPriority(-1)]
        public Moments moments(Mat src, bool binaryImage = false)
            => moments((InputArray)src, binaryImage);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="medianBlur(InputArray, int)"/>
        [OverloadResolutionPriority(-1)]
        public Mat medianBlur(Mat src, int kSize)
            => medianBlur((InputArray)src, kSize);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds NDArray arguments here through
        ///     the deferred implicit conversion, and the call reads those arrays, released at return.</summary>
        /// <inheritdoc cref="filter2D(InputArray, MatType, InputArray, Nullable{Point}, double, BorderTypes)"/>
        [OverloadResolutionPriority(-1)]
        public Mat filter2D(Mat src, MatType ddepth, Mat kernel, Point? anchor = null, double delta = 0, BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
            => filter2D((InputArray)src, ddepth, (InputArray)kernel, anchor, delta, borderType);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="blur(InputArray, Size, Nullable{Point}, BorderTypes)"/>
        [OverloadResolutionPriority(-1)]
        public Mat blur(Mat src, Size kSize, Point? anchor = null, BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
            => blur((InputArray)src, kSize, anchor, borderType);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="fastNlMeansDenoising(InputArray, float, int, int)"/>
        [OverloadResolutionPriority(-1)]
        public Mat fastNlMeansDenoising(Mat src, float h = 3, int templateWindowSize = 7, int searchWindowSize = 21)
            => fastNlMeansDenoising((InputArray)src, h, templateWindowSize, searchWindowSize);

        /// <summary>The pre-0.15 Mat overload, kept for source compatibility; forwards to the InputArray overload. C# 13+
        ///     prefers that overload for Mat and NDArray arguments; C# 12 binds an NDArray argument here through
        ///     the deferred implicit conversion, and the call reads that array itself, released at return.</summary>
        /// <inheritdoc cref="fastNlMeansDenoisingColored(InputArray, float, float, int, int)"/>
        [OverloadResolutionPriority(-1)]
        public Mat fastNlMeansDenoisingColored(Mat src, float h = 3, float hColor = 3, int templateWindowSize = 7, int searchWindowSize = 21)
            => fastNlMeansDenoisingColored((InputArray)src, h, hColor, templateWindowSize, searchWindowSize);
    }
}
