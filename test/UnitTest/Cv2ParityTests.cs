using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;
using SharpCV;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>
    ///     The rest of SharpCV's cv2 surface against opencv-python 4.13 on the same input arrays, bit for bit: filters,
    ///     every morphology operation with cv2's default border, structuring elements, contours and their drawing, text,
    ///     moments, denoising and channel splitting.
    /// </summary>
    [TestClass]
    public class Cv2ParityTests : Test
    {
        /// <summary>split gives one plane per channel, each equal to cv2's (and to the NumSharp channel view).</summary>
        [TestMethod]
        public void split()
        {
            var img = cv2.imread(imgSolar);
            var planes = cv2.split(img);
            Assert.AreEqual(3, planes.Length);
            NDArray src = img;
            for (int c = 0; c < 3; c++)
                Assert.IsTrue(np.array_equal(src[$":, :, {c}"], planes[c].data), $"plane {c} vs the NumSharp channel view");
            Oracle.Run(() =>
            {
                using var theirs = (PyObject)Oracle.cv2.split(src);
                for (int c = 0; c < 3; c++)
                    Oracle.AssertSame(theirs[c], planes[c].data, $"split plane {c}");
            });
        }

        /// <summary>The box filter and the median filter.</summary>
        [TestMethod]
        public void blur_and_medianBlur()
        {
            var img = cv2.imread(imgSolar);
            var boxed = cv2.blur(img, (5, 3));
            var median = cv2.medianBlur(img, 5);
            NDArray src = img;
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.blur(src, (5, 3)), boxed.data, "blur 5x3");
                Oracle.AssertSame(Oracle.cv2.medianBlur(src, 5), median.data, "medianBlur 5");
            });
        }

        /// <summary>Every morphology operation with cv2's defaults (anchor at the centre, the neutral border value), two
        ///     iterations. Until SharpCV 0.20 the default border value was 0, so erosions ate in from the image edge.</summary>
        [TestMethod]
        public void morphologyEx_every_operation_default_border()
        {
            var gray = cv2.cvtColor(cv2.imread(SmoothImage), ColorConversionCodes.COLOR_RGB2GRAY);
            var kernel = cv2.getStructuringElement(MorphShapes.MORPH_ELLIPSE, (5, 5));
            NDArray src = gray, k = kernel;
            foreach (MorphTypes op in new[] { MorphTypes.MORPH_ERODE, MorphTypes.MORPH_DILATE, MorphTypes.MORPH_OPEN, MorphTypes.MORPH_CLOSE,
                         MorphTypes.MORPH_GRADIENT, MorphTypes.MORPH_TOPHAT, MorphTypes.MORPH_BLACKHAT })
            {
                var mine = cv2.morphologyEx(gray, op, kernel, iterations: 2);
                Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.morphologyEx(src, (int)op, k, iterations: 2), mine.data, $"morphologyEx {op}"));
            }
        }

        /// <summary>Structuring elements with cv2's default anchor; MORPH_CROSS is the one the anchor shapes (SharpCV
        ///     anchored it at (0, 0) until 0.20, which drew an L).</summary>
        [TestMethod]
        public void getStructuringElement_default_anchor()
        {
            foreach (MorphShapes shape in new[] { MorphShapes.MORPH_RECT, MorphShapes.MORPH_CROSS, MorphShapes.MORPH_ELLIPSE })
            {
                var mine = cv2.getStructuringElement(shape, (5, 7));
                var anchored = cv2.getStructuringElement(shape, (5, 7), new Point(1, 2));
                Oracle.Run(() =>
                {
                    Oracle.AssertSame(Oracle.cv2.getStructuringElement((int)shape, (5, 7)), mine.data, $"{shape}");
                    Oracle.AssertSame(Oracle.cv2.getStructuringElement((int)shape, (5, 7), (1, 2)), anchored.data, $"{shape} anchor (1, 2)");
                });
            }
        }

        /// <summary>
        ///     findContours on the invoice's line mask, then drawContours three ways (the Mat list cv2 returns, managed
        ///     points, one contour at a time): every drawing equals cv2's, and the managed points equal cv2's contours.
        /// </summary>
        [TestMethod]
        public void contours_found_and_drawn()
        {
            var image = cv2.imread(Path("invoice.jpg"));
            var gray = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2GRAY);
            var (_, thresh) = cv2.threshold(gray, 0, 255, ThresholdTypes.THRESH_BINARY_INV | ThresholdTypes.THRESH_OTSU);
            var horizontal = cv2.getStructuringElement(MorphShapes.MORPH_RECT, (20, 1));
            var lines = cv2.morphologyEx(thresh, MorphTypes.MORPH_OPEN, horizontal, iterations: 2);

            var (contours, hierarchy) = cv2.findContours(lines, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);
            var points = cv2.findContoursAsPoints(lines, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_SIMPLE);
            Assert.AreEqual(contours.Length, points.Length);
            Assert.IsTrue(contours.Length > 0);

            NDArray before = image.data.copy();
            var byMats = image.data.copy();
            cv2.drawContours(byMats, contours, -1, (255, 255, 255), 5);
            var byPoints = image.data.copy();
            cv2.drawContours(byPoints, points, -1, (255, 255, 255), 5);
            var oneByOne = image.data.copy();
            foreach (var c in contours)
                cv2.drawContours(oneByOne, c, -1, (255, 255, 255), 5);

            NDArray linesNd = lines;
            Oracle.Run(() =>
            {
                dynamic found = Oracle.cv2.findContours(linesNd, (int)RetrievalModes.RETR_EXTERNAL, (int)ContourApproximationModes.CHAIN_APPROX_SIMPLE);
                var theirs = (PyObject)found[0];
                Assert.AreEqual(theirs.Length(), contours.Length, "contour count");
                for (int i = 0; i < contours.Length; i++)
                {
                    Oracle.AssertSame(theirs[i], contours[i].data, $"contour {i}");
                    var flat = np.array(points[i].SelectMany(p => new[] { p.X, p.Y }).ToArray()).reshape(-1, 1, 2);
                    Oracle.AssertSame(theirs[i], flat, $"managed points of contour {i}");
                }

                Oracle.AssertSame(found[1], hierarchy.data, "hierarchy");
                NDArray expected = Oracle.cv2.drawContours(before.copy(), theirs, -1, (255, 255, 255), 5);
                Oracle.AssertSame(expected, byMats, "drawContours(Mat[])");
                Oracle.AssertSame(expected, byPoints, "drawContours(Point[][])");
                Oracle.AssertSame(expected, oneByOne, "drawContours(Mat), one contour at a time");
            });
        }

        /// <summary>approxPolyDP, contourArea (both orientations) and moments of the largest contour equal cv2's, the
        ///     moments field by field (spatial, central and normalized).</summary>
        [TestMethod]
        public void contour_measures()
        {
            var gray = cv2.imread(UncorrectedImage, IMREAD_COLOR.IMREAD_GRAYSCALE);
            var (_, thresh) = cv2.threshold(gray, 0, 255, ThresholdTypes.THRESH_BINARY_INV | ThresholdTypes.THRESH_OTSU);
            var (contours, _) = cv2.findContours(thresh, RetrievalModes.RETR_EXTERNAL, ContourApproximationModes.CHAIN_APPROX_NONE);
            var largest = contours.OrderByDescending(c => c.size).First();
            var poly = cv2.approxPolyDP(largest, 5, true);
            double area = cv2.contourArea(largest);
            double signed = cv2.contourArea(largest, oriented: true);
            var m = cv2.moments(largest);
            var mImage = cv2.moments(thresh, binaryImage: true);

            NDArray contour = largest, image = thresh;
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.approxPolyDP(contour, 5, true), poly.data, "approxPolyDP");
                Assert.AreEqual((double)Oracle.cv2.contourArea(contour), area, "contourArea");
                Assert.AreEqual((double)Oracle.cv2.contourArea(contour, true), signed, "oriented contourArea");
                CompareMoments(Oracle.cv2.moments(contour), m, "contour moments");
                CompareMoments(Oracle.cv2.moments(image, true), mImage, "binary image moments");
            });
        }

        /// <summary>Text drawn on a Mat and on an NDArray, and its measured size.</summary>
        [TestMethod]
        public void putText_and_getTextSize()
        {
            var img = cv2.imread(imgSolar);
            NDArray before = img.data.copy();
            cv2.putText(img, "SharpCV 0.20", (20, 60), HersheyFonts.HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2, LineTypes.LINE_AA);
            var onArray = before.copy();
            cv2.putText(onArray, "NumSharp", (30, 200), HersheyFonts.HERSHEY_DUPLEX, 2.0, (255, 0, 255), 3);
            var size = cv2.getTextSize("SharpCV 0.20", HersheyFonts.HERSHEY_SIMPLEX, 1.5, 2, out int baseLine);

            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.putText(before.copy(), "SharpCV 0.20", (20, 60), (int)HersheyFonts.HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2, (int)LineTypes.LINE_AA),
                    img.data, "putText on a Mat");
                Oracle.AssertSame(Oracle.cv2.putText(before.copy(), "NumSharp", (30, 200), (int)HersheyFonts.HERSHEY_DUPLEX, 2.0, (255, 0, 255), 3),
                    onArray, "putText on an NDArray");
                dynamic measured = Oracle.cv2.getTextSize("SharpCV 0.20", (int)HersheyFonts.HERSHEY_SIMPLEX, 1.5, 2);
                Assert.AreEqual(((int)measured[0][0], (int)measured[0][1], (int)measured[1]), (size.Width, size.Height, baseLine), "getTextSize");
            });
        }

        /// <summary>Non-local means denoising, grayscale and colour, on a crop (the filter is slow).</summary>
        [TestMethod]
        public void fastNlMeansDenoising()
        {
            var img = cv2.imread(imgSolar)[(100, 228), (200, 328)];
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_BGR2GRAY);
            var denoised = cv2.fastNlMeansDenoising(gray, 7);
            var colored = cv2.fastNlMeansDenoisingColored(img, 5, 5);
            NDArray g = gray, c = img;
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.fastNlMeansDenoising(g, null, 7f, 7, 21), denoised.data, "fastNlMeansDenoising");
                Oracle.AssertSame(Oracle.cv2.fastNlMeansDenoisingColored(c, null, 5f, 5f, 7, 21), colored.data, "fastNlMeansDenoisingColored");
            });
        }

        /// <summary>blobFromImage with cv2's defaults (scale 1, the image's size, swapRB false).</summary>
        [TestMethod]
        public void blobFromImage_defaults()
        {
            var img = cv2.imread(img3x5);
            var blob = cv2.dnn.blobFromImage(img);
            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.dnn.blobFromImage(src), blob.data, "blobFromImage defaults"));
        }

        /// <summary>Compares SharpCV's moments with cv2's dictionary.</summary>
        /// <param name="expected">cv2.moments' dict.</param>
        /// <param name="actual">SharpCV's moments.</param>
        /// <param name="what">The case, for messages.</param>
        private static void CompareMoments(dynamic expected, Moments actual, string what)
        {
            var fields = new (string, double)[]
            {
                ("m00", actual.M00), ("m10", actual.M10), ("m01", actual.M01), ("m20", actual.M20), ("m11", actual.M11),
                ("m02", actual.M02), ("m30", actual.M30), ("m21", actual.M21), ("m12", actual.M12), ("m03", actual.M03),
                ("mu20", actual.Mu20), ("mu11", actual.Mu11), ("mu02", actual.Mu02), ("mu30", actual.Mu30), ("mu21", actual.Mu21),
                ("mu12", actual.Mu12), ("mu03", actual.Mu03), ("nu20", actual.Nu20), ("nu11", actual.Nu11), ("nu02", actual.Nu02),
                ("nu30", actual.Nu30), ("nu21", actual.Nu21), ("nu12", actual.Nu12), ("nu03", actual.Nu03),
            };
            foreach (var (name, value) in fields)
                Assert.AreEqual((double)expected[name], value, $"{what}: {name}");
        }

        /// <summary>A file in the data folder.</summary>
        /// <param name="name">The file name.</param>
        /// <returns>Its full path.</returns>
        private string Path(string name) => System.IO.Path.Combine(datadir, name);
    }
}
