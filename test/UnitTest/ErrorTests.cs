using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>
    ///     OpenCV errors surface as <see cref="OpenCVException"/> with cv2's code and message (they returned an empty Mat
    ///     silently before SharpCV 0.20), and the library keeps working after one.
    /// </summary>
    [TestClass]
    public class ErrorTests : Test
    {
        /// <summary>cvtColor BGR to GRAY on a single-channel image: OpenCV's channel assertion, the same code and
        ///     description as cv2.error.</summary>
        [TestMethod]
        public void opencv_error_has_cv2_code_and_message()
        {
            NDArray gray = cv2.imread(imgSolar, IMREAD_COLOR.IMREAD_GRAYSCALE);
            Oracle.Run(() =>
            {
                var e = Oracle.AssertBothFail(
                    () => cv2.cvtColor(gray, ColorConversionCodes.COLOR_BGR2GRAY),
                    () => Oracle.cv2.cvtColor(gray, (int)ColorConversionCodes.COLOR_BGR2GRAY),
                    "cvtColor BGR2GRAY on one channel");
                Assert.AreEqual(ErrorCode.BadNumChannels, e.Status);
                StringAssert.Contains(e.Message, "error: (-15:Bad number of channels)");
                StringAssert.Contains(e.Message, "Invalid number of channels in input image");
            });

            // The library is still usable after the failure.
            var ok = cv2.cvtColor(cv2.imread(imgSolar), ColorConversionCodes.COLOR_BGR2GRAY);
            Assert.AreEqual(299300, ok.size);
        }

        /// <summary>resize with neither a size nor factors fails as in cv2.</summary>
        [TestMethod]
        public void resize_without_size_fails()
        {
            NDArray img = cv2.imread(imgSolar);
            Oracle.Run(() => Oracle.AssertBothFail(
                () => cv2.resize(img, (0, 0)),
                () => Oracle.cv2.resize(img, (0, 0)),
                "resize to (0, 0)"));
        }

        /// <summary>An argument OpenCV asserts on (an even median aperture) fails with cv2's code.</summary>
        [TestMethod]
        public void invalid_aperture_is_reported()
        {
            NDArray img = cv2.imread(imgSolar);
            Oracle.Run(() => Oracle.AssertBothFail(
                () => cv2.medianBlur(img, 4),
                () => Oracle.cv2.medianBlur(img, 4),
                "medianBlur with an even aperture"));
        }

        /// <summary>The binding-level refusals cv2 makes before calling OpenCV (dtype, read-only output) carry
        ///     StsBadArg, as cv2's argument parser does.</summary>
        [TestMethod]
        public void conversion_refusals_are_bad_argument()
        {
            var boolean = np.zeros(new Shape(4, 4), np.dtype("bool"));
            var readOnly = np.broadcast_to(np.zeros(new Shape(4), np.uint8), new Shape(4, 4));
            Oracle.Run(() =>
            {
                Oracle.AssertBothFail(() => cv2.blur(boolean, (3, 3)), () => Oracle.cv2.blur(boolean, (3, 3)), "bool input");
                Oracle.AssertBothFail(() => cv2.rectangle(readOnly, (0, 0), (1, 1), 1), () => Oracle.cv2.rectangle(readOnly, (0, 0), (1, 1), 1), "read-only output");
            });
        }
    }
}
