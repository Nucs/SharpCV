using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpCV;
using System;
using NumSharp;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>The OCR enhancement pipeline (sharpen with a NumSharp kernel, then open), against opencv-python.</summary>
    [TestClass]
    public class OCRImageEnhance : Test
    {
        /// <summary>Reads the scan as grayscale.</summary>
        /// <returns>The grayscale image.</returns>
        public Mat imread()
        {
            var img = cv2.imread(SmoothImage);
            var gray = cv2.cvtColor(img, ColorConversionCodes.COLOR_RGB2GRAY);
            return gray;
        }

        /// <summary>Writes the enhanced image.</summary>
        /// <param name="img">The image.</param>
        public void imwrite(Mat img)
        {
            var result = cv2.imwrite("EnhancedImage.jpg", img);
            Assert.IsTrue(result);
        }

        /// <summary>filter2D with a float32 NDArray kernel, then MORPH_OPEN with cv2's default border.</summary>
        [TestMethod]
        public void enhanceImage()
        {
            Mat img = imread();

            NDArray filter1_kernel = np.array(new float[3, 3] { { 0, -1, 0 }, { -1, 5, -1 }, { 0, -1, 0 } });
            Mat filter1 = cv2.filter2D(img, MatType.CV_8UC1, new Mat(filter1_kernel), new Point(-1, -1), 0);
            var open_kernel = cv2.getStructuringElement(MorphShapes.MORPH_RECT, (3, 3));
            var img_open = cv2.morphologyEx(filter1, MorphTypes.MORPH_OPEN, open_kernel, iterations: 1);
            // var img_median = cv2.medianBlur(img_open, 3);

            imwrite(img_open);

            NDArray gray = img, filtered = filter1, kernel = open_kernel;
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.filter2D(gray, 0, filter1_kernel), filtered, "filter2D sharpen");
                Oracle.AssertSame(Oracle.cv2.morphologyEx(filtered, (int)MorphTypes.MORPH_OPEN, kernel, iterations: 1), img_open.data, "MORPH_OPEN");
            });
        }

    }
}
