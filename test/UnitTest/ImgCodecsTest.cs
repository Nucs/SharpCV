using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using System.IO;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>Decoding, encoding and cropping, against opencv-python.</summary>
    [TestClass]
    public class ImgCodecsTest : Test
    {
        /// <summary>imread decodes every test image to exactly cv2.imread's pixels (the same OpenCV, the same codecs).</summary>
        [TestMethod]
        public void imread()
        {
            var img = cv2.imread(imgSolar);
            Assert.AreEqual(897900, img.size);
            Assert.AreEqual((410, 730, 3), img.shape);

            foreach (var file in new[] { imgSolar, img3x5, UncorrectedImage, SmoothImage, Path.Combine(datadir, "invoice.jpg"), Path.Combine(datadir, "light.jpg") })
            {
                NDArray color = cv2.imread(file);
                NDArray gray = cv2.imread(file, IMREAD_COLOR.IMREAD_GRAYSCALE);
                Oracle.Run(() =>
                {
                    Oracle.AssertSame(Oracle.cv2.imread(file), color, $"imread {Path.GetFileName(file)}");
                    Oracle.AssertSame(Oracle.cv2.imread(file, (int)IMREAD_COLOR.IMREAD_GRAYSCALE), gray, $"imread gray {Path.GetFileName(file)}");
                });
            }
        }

        /// <summary>A missing file gives an empty Mat whose data is null, as cv2.imread gives None.</summary>
        [TestMethod]
        public void imread_missing_file_is_empty()
        {
            var img = cv2.imread(Path.Combine(datadir, "no-such-image.png"));
            Assert.IsNull(img.data);
            Assert.AreEqual(0, img.size);
            Oracle.Run(() => Assert.IsNull((object)Oracle.cv2.imread(Path.Combine(datadir, "no-such-image.png")), "cv2 returns None"));
        }

        /// <summary>imwrite succeeds.</summary>
        [TestMethod]
        public void imwrite()
        {
            var img = cv2.imread(imgSolar);
            var result = cv2.imwrite("solar.jpg", img);
            Assert.IsTrue(result);
        }

        /// <summary>Both crop forms give the region cv2's slice <c>img[50:400, 150:350]</c> gives. Until SharpCV 0.20 the
        ///     region's NDArray ignored the parent's row step, so every row after the first was read from the wrong
        ///     place (row 1, column 0 read 201; cv2 says 86).</summary>
        [TestMethod]
        public void imcrop()
        {
            var img = cv2.imread(imgSolar);
            var cropped1 = cv2.imcrop(img, (150, 50, 200, 350));
            Assert.AreEqual((350, 200, 3), cropped1.shape);

            var cropped2 = img[(50, 400), (150, 350)];
            Assert.AreEqual((350, 200, 3), cropped2.shape);
            Assert.AreEqual(86, cropped2[1, 0, 0]);

            NDArray full = img;
            Oracle.Run(() =>
            {
                var expected = Oracle.at(Oracle.cv2.imread(imgSolar), Oracle.builtins.slice(50, 400), Oracle.builtins.slice(150, 350));
                Oracle.AssertSame(expected, cropped1.data, "imcrop");
                Oracle.AssertSame(expected, cropped2.data, "ROI indexer");
                Oracle.AssertSame(expected, full["50:400, 150:350"], "NumSharp slice of the image view");
            });
        }

        /// <summary>A PNG round trip through the byte[] API is lossless.</summary>
		[TestMethod]
		public void imendecode()
		{
            var mat = cv2.imread(imgSolar);
            var fbs_png = mat.ImEncode();
            var mat2 = SharpCV.Mat.ImDecode(fbs_png);

            CollectionAssert.AreEqual(mat.data.ToArray<byte>(), mat2.data.ToArray<byte>());
		}

        /// <summary>cv2.imencode returns a zero-copy uint8 view of OpenCV's buffer whose bytes equal cv2.imencode's, and
        ///     cv2.imdecode reads a NumSharp buffer directly.</summary>
        [TestMethod]
        public void imencode_imdecode_ndarray()
        {
            var mat = cv2.imread(imgSolar);
            var (ok, png) = cv2.imencode(".png", mat);
            Assert.IsTrue(ok);
            Assert.AreEqual(1, png.ndim);
            Assert.AreEqual(NPTypeCode.Byte, png.typecode);

            var decoded = cv2.imdecode(png, IMREAD_COLOR.IMREAD_COLOR);
            Assert.IsTrue(np.array_equal(mat.data, decoded.data));

            NDArray img = mat;
            Oracle.Run(() =>
            {
                dynamic encoded = Oracle.cv2.imencode(".png", img);
                Oracle.AssertSame((NDArray)encoded[1], png, "imencode .png bytes");
                Oracle.AssertSame(Oracle.cv2.imdecode(png, (int)IMREAD_COLOR.IMREAD_COLOR), decoded.data, "imdecode");
            });
        }
    }
}
