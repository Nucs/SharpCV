using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp;

namespace SharpCV
{
    public partial class cv_api
    {
        /// <summary><c>cv2.cvtColor(src, code, dstCn)</c>: converts between colour spaces (BGR to GRAY, ...).</summary>
        /// <param name="img">The image (a Mat or an NDArray; a <c>[..., ::-1]</c> channel-reversed view is copied
        ///     first, as cv2 does, and the copy is freed when the call returns).</param>
        /// <param name="code">The conversion.</param>
        /// <param name="dstCnt">The destination channel count; 0 derives it from the code.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image's channel count or depth does not fit the code, or cv2 refuses the
        ///     NDArray.</exception>
        public Mat cvtColor(InputArray img,
            ColorConversionCodes code,
            int dstCnt = 0)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_cvtColor(src.Mat.InputArray, dst.OutputArray, (int)code, dstCnt));
            return dst.Tracked();
        }

        /// <summary><c>cv2.contourArea(contour, oriented)</c>.</summary>
        /// <param name="img">The contour: a point set such as one of <see cref="findContours(InputArray, RetrievalModes, ContourApproximationModes, Point)"/>'s results.</param>
        /// <param name="oriented">True returns a signed area (its sign gives the orientation).</param>
        /// <returns>The area.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The input is not a point set, or cv2 refuses the NDArray.</exception>
        public double contourArea(InputArray img,
            bool oriented = false)
        {
            using var contour = img.Open("contour");
            double output = 0;
            NativeMethods.Call(() => cv2_native_api.imgproc_contourArea_InputArray(contour.Mat.InputArray, oriented ? 1 : 0, out output));
            return output;
        }

        /// <summary><c>cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)</c>.</summary>
        /// <param name="img">An 8-bit single-channel image.</param>
        /// <param name="maxValue">The value of pixels that pass.</param>
        /// <param name="method">Mean or Gaussian neighbourhood.</param>
        /// <param name="type">BINARY or BINARY_INV.</param>
        /// <param name="blockSize">The neighbourhood size (odd, at least 3).</param>
        /// <param name="delta">The constant subtracted from the neighbourhood mean (cv2's <c>C</c>).</param>
        /// <returns>A new binary image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is not 8-bit single-channel or the block size is even, or cv2
        ///     refuses the NDArray.</exception>
        public Mat adaptiveThreshold(InputArray img,
            double maxValue,
            AdaptiveThresholdTypes method,
            ThresholdTypes type,
            int blockSize,
            double delta)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_adaptiveThreshold(src.Mat.InputArray,
                dst.OutputArray,
                maxValue,
                (int)method,
                (int)type,
                blockSize,
                delta));
            return dst.Tracked();
        }

        /// <summary><c>cv2.threshold(src, thresh, maxval, type)</c>.</summary>
        /// <param name="img">The image.</param>
        /// <param name="thresh">The threshold (ignored with OTSU or TRIANGLE, which compute it).</param>
        /// <param name="maxval">The value of pixels that pass (BINARY types).</param>
        /// <param name="type">The threshold type, optionally combined with OTSU or TRIANGLE.</param>
        /// <returns>(retval, dst) as cv2 returns them: the threshold used, and the new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OTSU / TRIANGLE on an image that is not 8-bit single-channel, or cv2 refuses
        ///     the NDArray.</exception>
        public (double, Mat) threshold(InputArray img, double thresh, double maxval, ThresholdTypes type)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            double ret = 0;
            NativeMethods.Call(() => cv2_native_api.imgproc_threshold(src.Mat.InputArray, dst.OutputArray, thresh, maxval, (int)type, out ret));
            return (ret, dst.Tracked());
        }

        /// <summary><c>cv2.pyrUp(src)</c>: upsamples by two and blurs.</summary>
        /// <param name="img">The image.</param>
        /// <returns>A new image twice the size.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Mat pyrUp(InputArray img)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_pyrUp(src.Mat.InputArray,
                dst.OutputArray,
                new Size(),
                (int)BorderTypes.BORDER_DEFAULT));
            return dst.Tracked();
        }

        /// <summary><c>cv2.pyrUp(src, dstsize, borderType)</c>.</summary>
        /// <param name="img">The image.</param>
        /// <param name="dstSize">The output size (width, height), about twice the input's.</param>
        /// <param name="borderType">The border mode.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The size is not within one pixel of twice the input's, or cv2 refuses the
        ///     NDArray.</exception>
        public Mat pyrUp(InputArray img,
            (int, int) dstSize,
            BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_pyrUp(src.Mat.InputArray,
                dst.OutputArray,
                new Size(dstSize.Item1, dstSize.Item2),
                (int)borderType));
            return dst.Tracked();
        }

        /// <summary><c>cv2.pyrDown(src, dstsize, borderType)</c>.</summary>
        /// <param name="img">The image.</param>
        /// <param name="dstSize">The output size (width, height), about half the input's.</param>
        /// <param name="borderType">The border mode.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The size is not within one pixel of half the input's, or cv2 refuses the
        ///     NDArray.</exception>
        public Mat pyrDown(InputArray img,
            (int, int) dstSize,
            BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_pyrDown(src.Mat.InputArray,
                dst.OutputArray,
                new Size(dstSize.Item1, dstSize.Item2),
                (int)borderType));
            return dst.Tracked();
        }

        /// <summary><c>cv2.pyrDown(src)</c>: blurs and downsamples by two (rounding the size up).</summary>
        /// <param name="img">The image.</param>
        /// <returns>A new image half the size.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Mat pyrDown(InputArray img)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_pyrDown(src.Mat.InputArray,
                dst.OutputArray,
                new Size(),
                (int)BorderTypes.BORDER_DEFAULT));
            return dst.Tracked();
        }

        /// <summary><c>cv2.resize(src, dsize, fx, fy, interpolation)</c>.</summary>
        /// <param name="img">The image.</param>
        /// <param name="dsize">The output size as (width, height); (0, 0) derives it from the factors.</param>
        /// <param name="fx">The horizontal factor when <paramref name="dsize"/> is (0, 0).</param>
        /// <param name="fy">The vertical factor when <paramref name="dsize"/> is (0, 0).</param>
        /// <param name="interpolation">The interpolation method (INTER_AREA for shrinking).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">Both the size and the factors are zero, or cv2 refuses the NDArray.</exception>
        public Mat resize(InputArray img,
            (int, int) dsize,
            double fx = 0,
            double fy = 0,
            InterpolationFlags interpolation = InterpolationFlags.INTER_LINEAR)
        {
            using var src = img.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_resize(src.Mat.InputArray,
                dst.OutputArray,
                new Size(dsize.Item1, dsize.Item2),
                fx,
                fy,
                (int)interpolation));

            return dst.Tracked();
        }

        /// <summary>
        /// Using rotation matrix to rotate a image.
        /// https://en.wikipedia.org/wiki/Rotation_matrix
        /// </summary>
        /// <param name="img">The image.</param>
        /// <param name="center">The rotation centre as (x, y) = (column, row); integer and float tuples both convert.
        ///     Until SharpCV 0.15 the centre reached OpenCV as garbage floats near 5e-43, so every rotation turned about
        ///     the top-left corner.</param>
        /// <param name="angle">The angle in degrees, counter-clockwise.</param>
        /// <param name="scale">The isotropic scale.</param>
        /// <param name="flags">The interpolation.</param>
        /// <param name="borderMode">The border mode for pixels from outside the image.</param>
        /// <returns>A new image of the same size: <c>cv2.warpAffine(img, cv2.getRotationMatrix2D(center, angle, scale),
        ///     (w, h), flags, borderMode)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Mat rotate(InputArray img,
            Point2f center,
            double angle,
            double scale,
            InterpolationFlags flags = InterpolationFlags.INTER_LINEAR,
            BorderTypes borderMode = BorderTypes.BORDER_CONSTANT)
        {
            using var src = img.Open("src");
            using var matrix2d = getRotationMatrix2D(center, angle, scale);

            Size size = default;
            IntPtr source = src.Mat.Handle;
            NativeMethods.Call(() => cv2_native_api.core_Mat_size(source, out size));

            var scalar = new Scalar();
            var dst = new Mat();

            NativeMethods.Call(() => cv2_native_api.imgproc_warpAffine(src.Mat.InputArray,
                dst.OutputArray,
                matrix2d.InputArray,
                size,
                (int)flags,
                (int)borderMode, scalar));

            return dst.Tracked();
        }

        /// <summary><c>cv2.getRotationMatrix2D(center, angle, scale)</c>: the 2 x 3 float64 affine matrix of a rotation
        ///     about <paramref name="center"/>.</summary>
        /// <param name="center">The centre (x, y).</param>
        /// <param name="angle">The angle in degrees, counter-clockwise.</param>
        /// <param name="scale">The isotropic scale.</param>
        /// <returns>A new 2 x 3 CV_64FC1 Mat.</returns>
        /// <exception cref="OpenCVException">Never in practice.</exception>
        public Mat getRotationMatrix2D(Point2f center, double angle, double scale)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.imgproc_getRotationMatrix2D(center, angle, scale, out handle));
            return new Mat(handle);
        }

        /// <summary><c>cv2.warpAffine(src, M, dsize, flags, borderMode, borderValue)</c>.</summary>
        /// <param name="src">The image.</param>
        /// <param name="M">The 2 x 3 transformation; an NDArray (float32 or float64) converts implicitly.</param>
        /// <param name="dsize">The output size (width, height).</param>
        /// <param name="flags">The interpolation, optionally with WARP_INVERSE_MAP.</param>
        /// <param name="borderMode">The border mode.</param>
        /// <param name="borderValue">The value of a constant border; null is 0, cv2's default.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> or <paramref name="M"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The matrix is not 2 x 3, or cv2 refuses an NDArray.</exception>
        public Mat warpAffine(InputArray src,
            InputArray M,
            Size dsize,
            InterpolationFlags flags = InterpolationFlags.INTER_LINEAR,
            BorderTypes borderMode = BorderTypes.BORDER_CONSTANT,
            Scalar? borderValue = null)
        {
            using var image = src.Open("src");
            using var matrix = M.Open("M");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_warpAffine(image.Mat.InputArray, dst.OutputArray, matrix.Mat.InputArray,
                dsize, (int)flags, (int)borderMode, borderValue ?? new Scalar()));
            return dst.Tracked();
        }

        /// <summary><c>cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)</c>: draws INTO the image.</summary>
        /// <param name="img">The image, modified in place: a Mat, or an NDArray that is writeable and viewable as a Mat
        ///     (a column-strided or reversed view is refused, as cv2 refuses it, since drawing on a copy would be
        ///     lost).</param>
        /// <param name="leftTop">One corner.</param>
        /// <param name="rightBottom">The opposite corner.</param>
        /// <param name="color">The colour (BGR for a colour image).</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="type">The line type.</param>
        /// <param name="shift">Fractional bits in the coordinates.</param>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is (or the Mat views) a read-only array, an NDArray's dtype has no
        ///     depth or its layout would need a copy ("Layout of the output array img is incompatible with cv::Mat"), or
        ///     OpenCV rejects the input.</exception>
        public void rectangle(InputOutputArray img,
            Point leftTop,
            Point rightBottom,
            Scalar color,
            int thickness = 1,
            LineTypes type = LineTypes.LINE_8,
            int shift = 0)
        {
            using var dst = img.Open("img");
            NativeMethods.Call(() => cv2_native_api.imgproc_rectangle_InputOutputArray_Point(dst.Mat.OutputArray,
                leftTop,
                rightBottom,
                color,
                thickness,
                type,
                shift));
        }

        /// <summary><c>cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType,
        ///     bottomLeftOrigin)</c>: draws INTO the image.</summary>
        /// <param name="img">The image, modified in place: a Mat, or a writeable NDArray viewable as a Mat.</param>
        /// <param name="text">The text (Hershey fonts draw ASCII).</param>
        /// <param name="org">The bottom-left corner of the text.</param>
        /// <param name="fontFace">The font.</param>
        /// <param name="fontScale">The font scale.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The stroke thickness.</param>
        /// <param name="lineType">The line type.</param>
        /// <param name="bottomLeftOrigin">True when the image origin is the bottom-left corner.</param>
        /// <exception cref="ArgumentNullException"><paramref name="img"/> or <paramref name="text"/> is null or empty.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is (or the Mat views) a read-only array, an NDArray's dtype has no
        ///     depth or its layout would need a copy, or OpenCV rejects the input.</exception>
        public void putText(InputOutputArray img, string text, Point org,
            HersheyFonts fontFace, double fontScale, Scalar color,
            int thickness = 1, LineTypes lineType = LineTypes.LINE_8, bool bottomLeftOrigin = false)
        {
            // Both arguments are checked before the image is opened, so a refused call never builds a view of an NDArray.
            if (img.Mat is null && img.Array is null)
                throw new ArgumentNullException(nameof(img));
            if (string.IsNullOrEmpty(text))
                throw new ArgumentNullException(nameof(text));

            using var dst = img.Open("img");
            // putText writes the image: it takes the InputOutputArray (SharpCV passed the read-only InputArray until 0.15).
            NativeMethods.Call(() => cv2_native_api.imgproc_putText(dst.Mat.OutputArray, text, org, (int)fontFace, fontScale, color,
                thickness, (int)lineType, bottomLeftOrigin ? 1 : 0));
        }

        /// <summary><c>cv2.getTextSize(text, fontFace, fontScale, thickness)</c> with the baseline.</summary>
        /// <param name="text">The text.</param>
        /// <param name="fontFace">The font.</param>
        /// <param name="fontScale">The font scale.</param>
        /// <param name="thickness">The stroke thickness.</param>
        /// <param name="baseLine">Receives the baseline's y offset from the bottom of the text.</param>
        /// <returns>The text box size.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="text"/> is null or empty.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the font.</exception>
        public Size getTextSize(string text, HersheyFonts fontFace,
            double fontScale, int thickness, out int baseLine)
        {
            if (string.IsNullOrEmpty(text))
                throw new ArgumentNullException(nameof(text));

            int line = 0;
            Size ret = default;
            NativeMethods.Call(() => cv2_native_api.imgproc_getTextSize(text, (int)fontFace, fontScale, thickness, out line, out ret));
            baseLine = line;
            return ret;
        }

        /// <summary><c>cv2.getTextSize(text, fontFace, fontScale, thickness)</c> without the baseline.</summary>
        /// <param name="text">The text.</param>
        /// <param name="fontFace">The font.</param>
        /// <param name="fontScale">The font scale.</param>
        /// <param name="thickness">The stroke thickness.</param>
        /// <returns>The text box size.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="text"/> is null or empty.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the font.</exception>
        public Size getTextSize(string text, HersheyFonts fontFace, double fontScale, int thickness)
        {
            return getTextSize(text, fontFace, fontScale, thickness, out _);
        }

        /// <summary><c>cv2.getStructuringElement(shape, ksize, anchor)</c>.</summary>
        /// <param name="shape">Rectangle, cross or ellipse.</param>
        /// <param name="ksize">The kernel size (width, height).</param>
        /// <param name="anchor">The anchor; null is (-1, -1), the centre, as in cv2. It shapes MORPH_CROSS only: SharpCV
        ///     passed (0, 0) until 0.15, which drew the cross through the top-left corner.</param>
        /// <returns>A new 8-bit kernel.</returns>
        /// <exception cref="OpenCVException">The anchor lies outside the kernel.</exception>
        public Mat getStructuringElement(MorphShapes shape, Size ksize, Point? anchor = null)
        {
            IntPtr handle = IntPtr.Zero;
            NativeMethods.Call(() => cv2_native_api.imgproc_getStructuringElement((int)shape, ksize, anchor ?? new Point(-1, -1), out handle));
            return new Mat(handle);
        }

        /// <summary><c>cv2.morphologyEx(src, op, kernel, anchor, iterations, borderType, borderValue)</c>.</summary>
        /// <param name="src">The image.</param>
        /// <param name="type">The operation (open, close, gradient, ...).</param>
        /// <param name="kernel">The structuring element.</param>
        /// <param name="anchor">The anchor; null is the kernel centre.</param>
        /// <param name="iterations">How many times erosion and dilation run.</param>
        /// <param name="borderType">The border mode.</param>
        /// <param name="borderValue">The constant border's value; null is <c>cv2.morphologyDefaultBorderValue()</c> (all
        ///     components DBL_MAX), which makes the border neutral: an erosion does not eat in from the image edge. SharpCV
        ///     passed 0 until 0.15 (7,024 pixels of a 1040 x 720 two-pass erosion differed from cv2).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> or <paramref name="kernel"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses an NDArray.</exception>
        public Mat morphologyEx(InputArray src,
            MorphTypes type,
            InputArray kernel,
            Point? anchor = null,
            int iterations = 1,
            BorderTypes borderType = BorderTypes.BORDER_CONSTANT,
            Scalar? borderValue = null)
        {
            using var image = src.Open("src");
            using var element = kernel.Open("kernel");
            var output = new Mat();
            if (anchor is null)
                anchor = new Point(-1, -1);
            NativeMethods.Call(() => cv2_native_api.imgproc_morphologyEx(image.Mat.InputArray,
                output.OutputArray,
                (int)type,
                element.Mat.InputArray,
                anchor.Value,
                iterations,
                (int)borderType,
                borderValue ?? Scalar.MorphologyDefaultBorderValue));
            return output.Tracked();
        }

        /// <summary>
        ///     <c>cv2.findContours(image, mode, method, offset)</c> with every contour as managed points (the hierarchy is
        ///     discarded).
        /// </summary>
        /// <param name="src">An 8-bit binary image.</param>
        /// <param name="mode">The retrieval mode.</param>
        /// <param name="method">The approximation method.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>One point array per contour.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is not 8-bit single-channel (or 32-bit labels with
        ///     RETR_CCOMP / RETR_FLOODFILL), or cv2 refuses the NDArray.</exception>
        public Point[][] findContoursAsPoints(InputArray src,
            RetrievalModes mode,
            ContourApproximationModes method,
            Point offset = default)
        {
            using var image = src.Open("image");
            // OpenCvSharpExtern 4.13 fills vectors the caller allocates (4.4 allocated them itself).
            using var contoursVec = new VectorOfVectorPoint();
            IntPtr hierarchy = cv2_native_api.vector_Vec4i_new1();
            try
            {
                NativeMethods.Call(() => cv2_native_api.imgproc_findContours1_vector(image.Mat.InputArray,
                    contoursVec,
                    hierarchy,
                    (int)mode,
                    (int)method,
                    offset));
                return contoursVec.ToArray();
            }
            finally
            {
                cv2_native_api.vector_Vec4i_delete(hierarchy);
            }
        }

        /// <summary>
        ///     <c>cv2.findContours(image, mode, method, offset)</c>: (contours, hierarchy) as cv2 returns them. Each contour
        ///     is an N x 1 CV_32SC2 Mat (an <c>(N, 1, 2)</c> int32 array), the hierarchy a 1 x N CV_32SC4 Mat
        ///     (<c>(1, N, 4)</c>), empty when there is no contour (cv2 returns <c>None</c>).
        /// </summary>
        /// <param name="src">An 8-bit binary image.</param>
        /// <param name="mode">The retrieval mode.</param>
        /// <param name="method">The approximation method.</param>
        /// <param name="offset">Added to every point.</param>
        /// <returns>(contours, hierarchy).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is not 8-bit single-channel, or cv2 refuses the NDArray.</exception>
        public (Mat[], Mat) findContours(InputArray src,
            RetrievalModes mode,
            ContourApproximationModes method,
            Point offset = default)
        {
            using var image = src.Open("image");
            var hierarchy = new Mat();
            using (var contoursVec = new VectorOfMat())
            {
                NativeMethods.Call(() => cv2_native_api.imgproc_findContours1_OutputArray(image.Mat.InputArray,
                    contoursVec,
                    hierarchy.OutputArray,
                    (int)mode,
                    (int)method,
                    offset));
                return (contoursVec.ToArray(), hierarchy);
            }
        }

        /// <summary><c>cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, maxLevel=maxLevel,
        ///     offset=offset)</c> for contours given as managed points: draws INTO the image.</summary>
        /// <param name="image">The image, modified in place: a Mat, or a writeable NDArray viewable as a Mat.</param>
        /// <param name="contours">The contours.</param>
        /// <param name="contourIdx">The contour to draw; negative draws all.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The line type.</param>
        /// <param name="maxLevel">The deepest nesting level drawn (used with a hierarchy only).</param>
        /// <param name="offset">Added to every point.</param>
        /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="contours"/> is
        ///     null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is (or the Mat views) a read-only array, an NDArray's dtype has no
        ///     depth or its layout would need a copy, or the index is out of range.</exception>
        public void drawContours(InputOutputArray image,
            Point[][] contours,
            int contourIdx,
            Scalar color,
            int thickness = 1,
            LineTypes lineType = LineTypes.LINE_8,
            int maxLevel = int.MaxValue,
            Point offset = default)
        {
            if (contours is null)
                throw new ArgumentNullException(nameof(contours));
            using var dst = image.Open("image");
            var contourSize2 = contours.Select(pts => pts.Length).ToArray();
            using (var contoursPtr = new ArrayAddress2<Point>(contours))
            {
                NativeMethods.Call(() => cv2_native_api.imgproc_drawContours_vector(
                            dst.Mat.OutputArray, contoursPtr.Pointer, contours.Length, contourSize2,
                            contourIdx, color, thickness, (int)lineType, IntPtr.Zero, 0, maxLevel, offset));
            }
        }

        /// <summary><c>cv2.drawContours(image, [contour], contourIdx, ...)</c> for ONE contour: draws INTO the
        ///     image.</summary>
        /// <param name="image">The image, modified in place: a Mat, or a writeable NDArray viewable as a Mat.</param>
        /// <param name="contour">The contour (an N x 1 CV_32SC2 Mat, or an <c>(N, 1, 2)</c> int32 NDArray, read for the
        ///     call).</param>
        /// <param name="contourIdx">0 or negative to draw it.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The line type.</param>
        /// <param name="hierarchy">The hierarchy from <see cref="findContours(InputArray, RetrievalModes, ContourApproximationModes, Point)"/>, or null.</param>
        /// <param name="maxLevel">The deepest nesting level drawn.</param>
        /// <param name="offset">Added to every point; null is (0, 0), cv2's default (it was (-1, -1) until SharpCV 0.15,
        ///     which drew every contour one pixel up and left).</param>
        /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="contour"/> is
        ///     null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is (or the Mat views) a read-only array, an NDArray's dtype has no
        ///     depth or its layout would need a copy, cv2 refuses the contour NDArray, or OpenCV rejects the
        ///     contour.</exception>
        public void drawContours(InputOutputArray image,
            InputArray contour,
            int contourIdx,
            Scalar color,
            int thickness = 1,
            LineTypes lineType = LineTypes.LINE_8,
            Mat? hierarchy = null,
            int maxLevel = int.MaxValue,
            Point? offset = null)
        {
            using var points = contour.Open("contours");
            drawContours(image, new[] { points.Mat }, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);
        }

        /// <summary><c>cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel,
        ///     offset)</c> for the contour Mats <see cref="findContours(InputArray, RetrievalModes, ContourApproximationModes, Point)"/> returns: draws INTO the image.</summary>
        /// <param name="image">The image, modified in place: a Mat, or a writeable NDArray viewable as a Mat.</param>
        /// <param name="contours">The contours.</param>
        /// <param name="contourIdx">The contour to draw; negative draws all.</param>
        /// <param name="color">The colour.</param>
        /// <param name="thickness">The line thickness; negative fills.</param>
        /// <param name="lineType">The line type.</param>
        /// <param name="hierarchy">The hierarchy from <see cref="findContours(InputArray, RetrievalModes, ContourApproximationModes, Point)"/>, or null.</param>
        /// <param name="maxLevel">The deepest nesting level drawn.</param>
        /// <param name="offset">Added to every point; null is (0, 0).</param>
        /// <exception cref="ArgumentNullException"><paramref name="image"/>, <paramref name="contours"/> or one of its
        ///     contours is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The image is (or the Mat views) a read-only array, an NDArray's dtype has no
        ///     depth or its layout would need a copy, or OpenCV rejects a contour.</exception>
        public void drawContours(InputOutputArray image,
            Mat[] contours,
            int contourIdx,
            Scalar color,
            int thickness = 1,
            LineTypes lineType = LineTypes.LINE_8,
            Mat? hierarchy = null,
            int maxLevel = int.MaxValue,
            Point? offset = null)
        {
            if (contours is null)
                throw new ArgumentNullException(nameof(contours));
            if (offset is null)
                offset = new Point(0, 0);
            var contoursPtr = new IntPtr[contours.Length];
            for (int i = 0; i < contours.Length; i++)
            {
                // OpenCvSharpExtern dereferences every cv::Mat*: a null or disposed contour (IntPtr.Zero until SharpCV 0.15)
                // is an access violation .NET cannot catch, so both are refused here.
                if (contours[i] is null)
                    throw new ArgumentNullException(nameof(contours), $"contours[{i}] is null.");
                contoursPtr[i] = contours[i].Handle;
            }

            using var dst = image.Open("image");
            // The hierarchy parameter is an _InputArray; SharpCV passed the cv::Mat itself until 0.15.
            NativeMethods.Call(() => cv2_native_api.imgproc_drawContours_InputArray(
                        dst.Mat.OutputArray, contoursPtr, contoursPtr.Length,
                        contourIdx, color, thickness, lineType,
                        hierarchy is null ? IntPtr.Zero : hierarchy.InputArray, maxLevel, offset.Value));

            GC.KeepAlive(contours);
            GC.KeepAlive(hierarchy);
        }

        /// <summary><c>cv2.approxPolyDP(curve, epsilon, closed)</c>: a polygon with fewer vertices within
        ///     <paramref name="epsilon"/> of the curve.</summary>
        /// <param name="curve">The curve (a contour).</param>
        /// <param name="epsilon">The maximal distance between the curve and its approximation.</param>
        /// <param name="closed">True for a closed curve.</param>
        /// <returns>A new point Mat.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="curve"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The input is not a point set, or cv2 refuses the NDArray.</exception>
        public Mat approxPolyDP(InputArray curve, double epsilon, bool closed)
        {
            using var points = curve.Open("curve");
            var approxCurve = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_approxPolyDP_InputArray(points.Mat.InputArray,
                approxCurve.OutputArray,
                epsilon,
                closed));
            return approxCurve;
        }

        /// <summary><c>cv2.minAreaRect(points)</c>: the rotated rectangle of minimal area around the points.</summary>
        /// <param name="points">The points (a contour).</param>
        /// <returns>The rectangle (centre, size, angle in degrees).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="points"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The input is not a point set, or cv2 refuses the NDArray.</exception>
        public RotatedRect minAreaRect(InputArray points)
        {
            using var set = points.Open("points");
            RotatedRect rect = default;
            NativeMethods.Call(() => cv2_native_api.imgproc_minAreaRect_InputArray(set.Mat.InputArray, out rect));
            return rect;
        }

        /// <summary><c>cv2.moments(array, binaryImage)</c>: spatial, central and normalized central moments.</summary>
        /// <param name="src">An image or a contour.</param>
        /// <param name="binaryImage">True counts every non-zero pixel as 1.</param>
        /// <returns>The moments.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input, or cv2 refuses the NDArray.</exception>
        public Moments moments(InputArray src, bool binaryImage = false)
            => new Moments(src, binaryImage);

        /// <summary><c>cv2.medianBlur(src, ksize)</c>.</summary>
        /// <param name="src">The image.</param>
        /// <param name="kSize">The aperture (odd, greater than 1; at most 5 for depths other than 8-bit).</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">The aperture is even or too large for the depth, or cv2 refuses the
        ///     NDArray.</exception>
        public Mat medianBlur(InputArray src, int kSize)
        {
            using var image = src.Open("src");
            var dst = new Mat();
            NativeMethods.Call(() => cv2_native_api.imgproc_medianBlur(image.Mat.InputArray, dst.OutputArray, kSize));
            return dst.Tracked();
        }

        /// <summary><c>cv2.filter2D(src, ddepth, kernel, anchor, delta, borderType)</c>: correlates the image with the
        ///     kernel.</summary>
        /// <param name="src">The image.</param>
        /// <param name="ddepth">The result depth; <see cref="MatType.Unknown"/> (-1) keeps the source's, as cv2's -1.</param>
        /// <param name="kernel">The kernel (a Mat, or a float NDArray such as <c>np.array(new float[,] {...})</c>).</param>
        /// <param name="anchor">The anchor; null is the kernel centre.</param>
        /// <param name="delta">Added to every result.</param>
        /// <param name="borderType">The border mode.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> or <paramref name="kernel"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the depth combination, or cv2 refuses an NDArray.</exception>
        public Mat filter2D(InputArray src,
            MatType ddepth,
            InputArray kernel,
            Point? anchor = null,
            double delta = 0,
            BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
        {
            using var image = src.Open("src");
            using var weights = kernel.Open("kernel");
            var dst = new Mat();
            if (anchor is null)
                anchor = new Point(-1, -1);
            NativeMethods.Call(() => cv2_native_api.imgproc_filter2D(image.Mat.InputArray,
                dst.OutputArray,
                ddepth,
                weights.Mat.InputArray,
                anchor.Value,
                delta,
                borderType));
            return dst.Tracked();
        }

        /// <summary><c>cv2.blur(src, ksize, anchor, borderType)</c>: the normalized box filter.</summary>
        /// <param name="src">The image.</param>
        /// <param name="kSize">The box size (width, height).</param>
        /// <param name="anchor">The anchor; null is the box centre.</param>
        /// <param name="borderType">The border mode.</param>
        /// <returns>A new image.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="src"/> is null.</exception>
        /// <exception cref="ObjectDisposedException">A Mat argument was disposed.</exception>
        /// <exception cref="OpenCVException">OpenCV rejects the input (an n-D Mat), or cv2 refuses the NDArray.</exception>
        public Mat blur(InputArray src,
            Size kSize,
            Point? anchor = null,
            BorderTypes borderType = BorderTypes.BORDER_DEFAULT)
        {
            using var image = src.Open("src");
            var dst = new Mat();
            if (anchor is null)
                anchor = new Point(-1, -1);
            NativeMethods.Call(() => cv2_native_api.imgproc_blur(image.Mat.InputArray,
                dst.OutputArray,
                kSize,
                anchor.Value,
                borderType));
            return dst.Tracked();
        }
    }
}
