using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpCV;
using System;
using NumSharp;
using Python.Runtime;
using static SharpCV.Binding;


namespace UnitTest
{
    /// <summary>
    ///     The deskew pipeline (threshold, dilate, largest contour, polygon, minAreaRect, rotate), checked step by step
    ///     against opencv-python: each cv2 call gets SharpCV's previous result as its input, so a difference is pinned to
    ///     the step that made it.
    /// </summary>
    [TestClass]
    public class CorrectImageTest : Test
    {

        /// <summary>Reads the skewed scan as grayscale.</summary>
        /// <returns>The grayscale image.</returns>
        public Mat imread()
        {
            var img = cv2.imread(UncorrectedImage);
            //Assert.AreEqual(1760094, img.size);
            //Assert.AreEqual((854, 687, 3), img.shape);
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
            return gray;
        }

        /// <summary>Writes the corrected image.</summary>
        /// <param name="img">The image.</param>
        public void imwrite(Mat img)
        {
            var result = cv2.imwrite("CorrectedImage.png", img);
            Assert.IsTrue(result);
        }

        /// <summary>The whole pipeline, with the final rotation compared against cv2.warpAffine of
        ///     cv2.getRotationMatrix2D (the centre is the test's own (rows / 2, cols / 2), on both sides).</summary>
        [TestMethod]
        public void correctImage()
        {
            Mat img = imread();
            var angle = findAngle(img);
            Mat correctedImage = new Mat();
            var center = ((int)img.shape[0] / 2, (int)img.shape[1] / 2);
            correctedImage = cv2.rotate(img, center, angle, 1.0, InterpolationFlags.INTER_LINEAR, BorderTypes.BORDER_DEFAULT);
            imwrite(correctedImage);

            NDArray gray = img;
            Oracle.Run(() =>
            {
                dynamic M = Oracle.cv2.getRotationMatrix2D(center, (double)angle, 1.0);
                NDArray expected = Oracle.cv2.warpAffine(gray, M, ((int)gray.shape[1], (int)gray.shape[0]),
                    flags: (int)InterpolationFlags.INTER_LINEAR, borderMode: (int)BorderTypes.BORDER_DEFAULT);
                Oracle.AssertSame(expected, correctedImage.data, "deskew rotation");
            });
        }

        /// <summary>
        ///     Finds the skew angle: Otsu threshold, two dilations, the largest external contour, its polygon and the
        ///     angle of its minimal rectangle; every step is compared with cv2.
        /// </summary>
        /// <param name="img">The grayscale image.</param>
        /// <returns>The angle in degrees.</returns>
        public float findAngle(Mat img)
        {
            float angle = 0;
            var (_, thresh) = cv2.threshold(img, 0, 255, ThresholdTypes.THRESH_BINARY_INV | ThresholdTypes.THRESH_OTSU);
            var dilate_kernel = cv2.getStructuringElement(MorphShapes.MORPH_RECT, (3, 3));
            var dilate = cv2.morphologyEx(thresh, MorphTypes.MORPH_DILATE, dilate_kernel, iterations: 2);

            //var cnts = cv2.findContoursAsPoints(dilate, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_NONE);
            //Point[] max_contour = new Point[] { };
            //foreach (var c in cnts)//TODO: use length filter here,but area filter is better
            //    if (c.Length > max_contour.Length) max_contour = c;
            var (cnts, _) = cv2.findContours(dilate, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_NONE);
            Mat max_contour = new Mat();
            foreach (var c in cnts)//TODO: use length filter here,but area filter is better
                if (c.size > max_contour.size) max_contour = c;

            Mat contours_poly = cv2.approxPolyDP(max_contour, 30, true);
            RotatedRect rotatedRect = cv2.minAreaRect(contours_poly);
            angle = rotatedRect.Angle;
            if (Math.Abs(angle) > 45.0)
                angle = 90.0f + rotatedRect.Angle;

            //var color = cv2.cvtColor(img, ColorConversionCodes.COLOR_GRAY2RGB);
            //cv2.drawContours(color, new[] { max_contour }, -1, (255, 0, 0), 5);
            //cv2.imwrite("test1.jpg", color);

            NDArray gray = img, threshNd = thresh, kernelNd = dilate_kernel, dilateNd = dilate, contourNd = max_contour, polyNd = contours_poly;
            Oracle.Run(() =>
            {
                dynamic t = Oracle.cv2.threshold(gray, 0, 255, (int)(ThresholdTypes.THRESH_BINARY_INV | ThresholdTypes.THRESH_OTSU));
                Oracle.AssertSame((NDArray)t[1], threshNd, "Otsu threshold");
                Oracle.AssertSame(Oracle.cv2.getStructuringElement((int)MorphShapes.MORPH_RECT, (3, 3)), kernelNd, "structuring element");
                Oracle.AssertSame(Oracle.cv2.morphologyEx(threshNd, (int)MorphTypes.MORPH_DILATE, kernelNd, iterations: 2), dilateNd, "dilate x2");

                dynamic found = Oracle.cv2.findContours(dilateNd, (int)RetrievalModes.RETR_EXTERNAL, (int)ContourApproximationModes.CHAIN_APPROX_NONE);
                var contours = (PyObject)found[0];
                Assert.AreEqual(contours.Length(), cnts.Length, "contour count");
                for (int i = 0; i < cnts.Length; i++)
                    Oracle.AssertSame((NDArray)(dynamic)contours[i], cnts[i].data, $"contour {i}");

                Oracle.AssertSame(Oracle.cv2.approxPolyDP(contourNd, 30, true), polyNd, "approxPolyDP");
                dynamic rect = Oracle.cv2.minAreaRect(polyNd);
                Assert.AreEqual((float)(double)rect[2], rotatedRect.Angle, "minAreaRect angle");
            });
            return angle;
        }
    }
}
