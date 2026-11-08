using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using System.IO;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>imgproc functions, each compared bit for bit with opencv-python on the same input array.</summary>
    [TestClass]
    public class ImgProcTest : Test
    {
        /// <summary>RGB to GRAY.</summary>
        [TestMethod]
        public void cvtColor()
        {
            var img = cv2.imread(imgSolar);
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
            Assert.AreEqual(299300, gray.size);

            NDArray src = img;
            var hsv = cv2.cvtColor(img, ColorConversionCodes.COLOR_BGR2HSV);
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.cvtColor(src, (int)ColorConversionCodes.COLOR_RGB2GRAY), gray.data, "RGB2GRAY");
                Oracle.AssertSame(Oracle.cv2.cvtColor(src, (int)ColorConversionCodes.COLOR_BGR2HSV), hsv.data, "BGR2HSV");
            });
        }

        /// <summary>The TRIANGLE threshold picks 81, as cv2 4.13 does.</summary>
        [TestMethod]
        public void threshold()
        {
            var img = cv2.imread(imgSolar);
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
            var (ret, binary) = cv2.threshold(gray, 0, 255, ThresholdTypes.THRESH_BINARY | ThresholdTypes.THRESH_TRIANGLE);
            Assert.AreEqual(299300, binary.size);
            // python version is 80.0
            Assert.AreEqual(81, ret);

            NDArray src = gray;
            Oracle.Run(() =>
            {
                dynamic expected = Oracle.cv2.threshold(src, 0, 255, (int)(ThresholdTypes.THRESH_BINARY | ThresholdTypes.THRESH_TRIANGLE));
                Assert.AreEqual((double)expected[0], ret, "retval");
                Oracle.AssertSame((NDArray)expected[1], binary.data, "THRESH_TRIANGLE");
            });
        }

        /// <summary>Gaussian adaptive threshold.</summary>
        [TestMethod]
        public void adaptiveThreshold()
        {
            var img = cv2.imread(imgSolar);
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
            var binary = cv2.adaptiveThreshold(gray, 255, AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C, ThresholdTypes.THRESH_BINARY, 25, 10);
            Assert.AreEqual(299300, binary.size);

            NDArray src = gray;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.adaptiveThreshold(src, 255, (int)AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C,
                (int)ThresholdTypes.THRESH_BINARY, 25, 10), binary.data, "adaptiveThreshold"));
        }

        /// <summary>pyrDown halves the image (rounding up).</summary>
        [TestMethod]
        public void pyrDown()
        {
            var img = cv2.imread(imgSolar);
            var dst = cv2.pyrDown(img);
            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.pyrDown(src), dst.data, "pyrDown"));
        }

        /// <summary>pyrUp doubles the image.</summary>
        [TestMethod]
        public void pyrUp()
        {
            var img = cv2.imread(imgSolar);
            var dst = cv2.pyrUp(img);
            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.pyrUp(src), dst.data, "pyrUp"));
        }

        /// <summary>resize with every interpolation.</summary>
        [TestMethod]
        public void resize()
        {
            var img = cv2.imread(imgSolar);
            var r = 200.0 / img.shape[1];
            var size = (200, (int)(img.shape[0] * r));
            var resized = cv2.resize(img, size, interpolation: InterpolationFlags.INTER_AREA);
            Assert.AreEqual(112, resized.shape[0]);
            Assert.AreEqual(200, resized.shape[1]);
            Assert.AreEqual(3, resized.shape[2]);

            NDArray src = img;
            foreach (var interpolation in new[] { InterpolationFlags.INTER_AREA, InterpolationFlags.INTER_NEAREST, InterpolationFlags.INTER_LINEAR, InterpolationFlags.INTER_CUBIC, InterpolationFlags.INTER_LANCZOS4 })
            {
                var mine = cv2.resize(img, size, interpolation: interpolation);
                Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.resize(src, size, interpolation: (int)interpolation), mine.data, $"resize {interpolation}"));
            }
        }

        /// <summary>A 180 degree rotation about the centre: the image's own pixels, upside down (not the near-black image
        ///     SharpCV produced before 0.20, when the centre reached OpenCV as ~5e-43).</summary>
        [TestMethod]
        public void rotate()
        {
            var img = cv2.imread(imgSolar);
            var rotated = cv2.rotate(img, ((int)img.shape[1] / 2, (int)img.shape[0] / 2), 180, 1.0);
            Assert.AreEqual(410, rotated.shape[0]);
            Assert.AreEqual(730, rotated.shape[1]);
            Assert.AreEqual(3, rotated.shape[2]);
            Assert.IsTrue(np.count_nonzero(rotated.data) > 800_000, "the rotated image is not black");

            NDArray src = img;
            Oracle.Run(() =>
            {
                dynamic M = Oracle.cv2.getRotationMatrix2D((365, 205), 180.0, 1.0);
                Oracle.AssertSame(Oracle.cv2.warpAffine(src, M, (730, 410)), rotated.data, "rotate 180 about the centre");
                Oracle.AssertSame((NDArray)M, cv2.getRotationMatrix2D((365, 205), 180, 1.0).data, "getRotationMatrix2D");
            });
        }

        /// <summary>A rectangle drawn on a Mat; the same drawing by cv2 on a copy of the same pixels.</summary>
        [TestMethod]
        public void rectangle()
        {
            var img = cv2.imread(imgSolar);
            NDArray before = img.data.copy();
            cv2.rectangle(img, (0, 0), (img.shape[1] / 2, img.shape[0] / 2), (255, 0, 0));
            cv2.imwrite("rectangle.jpg", img);

            Oracle.Run(() =>
            {
                NDArray expected = Oracle.cv2.rectangle(before, (0, 0), (365, 205), (255, 0, 0));
                Oracle.AssertSame(expected, img.data, "rectangle");
            });
        }

        /// <summary>minAreaRect of the first contour, as cv2 4.13 computes it.</summary>
        [TestMethod]
        public void minAreaRect()
        {
            var img = cv2.imread(imgSolar, IMREAD_COLOR.IMREAD_GRAYSCALE);
            var (ret, thresh) = cv2.threshold(img, 127, 255, 0);
            var (contours, hierarchy) = cv2.findContours(thresh, RetrievalModes.RETR_LIST, ContourApproximationModes.CHAIN_APPROX_SIMPLE);
            var cnt = contours[0];
            var rect = cv2.minAreaRect(cnt);

            NDArray threshNd = thresh, cntNd = cnt;
            Oracle.Run(() =>
            {
                dynamic found = Oracle.cv2.findContours(threshNd, (int)RetrievalModes.RETR_LIST, (int)ContourApproximationModes.CHAIN_APPROX_SIMPLE);
                Oracle.AssertSame((NDArray)found[0][0], cntNd, "first contour");
                Oracle.AssertSame((NDArray)found[1], hierarchy.data, "hierarchy");
                dynamic expected = Oracle.cv2.minAreaRect(cntNd);
                Assert.AreEqual((float)(double)expected[2], rect.Angle, "angle");
                Assert.AreEqual(((float)(double)expected[1][0], (float)(double)expected[1][1]), (rect.Size.Width, rect.Size.Height), "size");
                Assert.AreEqual(((float)(double)expected[0][0], (float)(double)expected[0][1]), (rect.Center.X, rect.Center.Y), "center");
            });
        }
    }
}
