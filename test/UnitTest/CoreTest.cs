using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpCV;
using System;
using System.Collections.Generic;
using System.Text;
using static SharpCV.Binding;
using NumSharp;
using System.IO;
using System.Linq;

namespace UnitTest
{
    /// <summary>Pixel access, concatenation, rotation and flips, each checked against opencv-python.</summary>
    [TestClass]
    public class CoreTest : Test
    {
        /// <summary>A pixel read three ways (NDArray view, byte indexer, grayscale scalar) equals cv2's
        ///     <c>img[8, 8]</c>.</summary>
        [TestMethod]
        public void pixel_getter()
        {
            // getter
            var img1 = cv2.imread(imgSolar);
            var px = img1[8, 8];
            var (b, g, r) = ((byte)px[0], (byte)px[1], (byte)px[2]);
            Assert.AreEqual((32, 19, 11), (b, g, r));

            // access specific channel
            b = img1[8, 8, 0];
            g = img1[8, 8, 1];
            r = img1[8, 8, 2];
            Assert.AreEqual((32, 19, 11), (b, g, r));

            var img2 = cv2.imread(imgSolar, IMREAD_COLOR.IMREAD_GRAYSCALE);
            byte p = (byte)img2[8, 8];
            Assert.AreEqual(18, p);

            Oracle.Run(() =>
            {
                var expected = Oracle.at(Oracle.cv2.imread(imgSolar), 8, 8);
                Oracle.AssertSame(expected, px, "imread(solar)[8, 8]");
            });
        }

        /// <summary>Writing a pixel through the Mat's NDArray view changes the Mat's own memory.</summary>
        [TestMethod]
        public void pixel_setter()
        {
            var img = cv2.imread(imgSolar);
            img[8, 8] = np.array(new byte[] { 0, 0, 255 });
            Assert.AreEqual((0, 0, 255), (img[8, 8, 0], img[8, 8, 1], img[8, 8, 2]));

            // A scalar broadcasts over the channels, as numpy's img[8, 9] = 7 does.
            img[8, 9] = (NDArray)(byte)7;
            Assert.AreEqual((7, 7, 7), (img[8, 9, 0], img[8, 9, 1], img[8, 9, 2]));
        }

        /// <summary>cv2.hconcat of the image with itself.</summary>
        [TestMethod]
        public void hconcat()
        {
            var img = cv2.imread(imgSolar);
            var hconcated = cv2.hconcat(img, img);
            Assert.AreEqual(410, hconcated.shape[0]);
            Assert.AreEqual(1460, hconcated.shape[1]);
            Assert.AreEqual(3, hconcated.shape[2]);

            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.hconcat(Oracle.list(src, src)), hconcated.data, "hconcat"));
        }

        /// <summary>
        ///     cv2.hconcat / vconcat of a Mat next to an NDArray that needs a copy (each argument converts on its own),
        ///     and of a <c>Mat[]</c> variable (which the <c>params</c> overload of proxies cannot take).
        /// </summary>
        [TestMethod]
        public void concat_mixed_arguments_and_mat_array()
        {
            using var img = cv2.imread(imgSolar);
            NDArray src = img;
            var reversed = src["..., ::-1"];
            using var mixed = cv2.hconcat(img, reversed);
            Mat[] both = { img, img };
            using var fromArray = cv2.vconcat(both);

            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.hconcat(Oracle.list(src, reversed)), mixed.data, "hconcat Mat + reversed NDArray");
                Oracle.AssertSame(Oracle.cv2.vconcat(Oracle.list(src, src)), fromArray.data, "vconcat Mat[]");
            });
        }

        /// <summary>
        ///     A null or disposed Mat among the images is refused before OpenCvSharpExtern dereferences it: until SharpCV
        ///     0.15 it crossed as a null <c>cv::Mat*</c>, an access violation that ends the process.
        /// </summary>
        [TestMethod]
        public void concat_refuses_null_and_disposed_mats()
        {
            using var img = cv2.imread(imgSolar);
            var disposed = cv2.imread(imgSolar);
            disposed.Dispose();

            Assert.ThrowsException<ArgumentNullException>(() => cv2.hconcat(new Mat[] { img, null }));
            Assert.ThrowsException<ObjectDisposedException>(() => cv2.vconcat(new Mat[] { img, disposed }));
            Assert.ThrowsException<ObjectDisposedException>(() => cv2.hconcat(img, disposed));
        }

        /// <summary>cv2.vconcat of the image with itself.</summary>
        [TestMethod]
        public void vconcat()
        {
            var img = cv2.imread(imgSolar);
            var hconcated = cv2.vconcat(img, img);
            Assert.AreEqual(820, hconcated.shape[0]);
            Assert.AreEqual(730, hconcated.shape[1]);
            Assert.AreEqual(3, hconcated.shape[2]);

            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.vconcat(Oracle.list(src, src)), hconcated.data, "vconcat"));
        }

        /// <summary>cv2.rotate by 90 degrees.</summary>
        [TestMethod]
        public void rotate()
        {
            var img = cv2.imread(imgSolar);
            var rotated = cv2.rotate(img, RotateFlags.ROTATE_90_CLOCKWISE);
            Assert.AreEqual(730, rotated.shape[0]);
            Assert.AreEqual(410, rotated.shape[1]);
            Assert.AreEqual(3, rotated.shape[2]);

            NDArray src = img;
            Oracle.Run(() => Oracle.AssertSame(Oracle.cv2.rotate(src, (int)RotateFlags.ROTATE_90_CLOCKWISE), rotated.data, "rotate 90"));
        }

        /// <summary>Every FLIP_MODE matches cv2's flip code: horizontal 1, vertical 0, both -1 (SharpCV sent 2 and 3,
        ///     both horizontal, until 0.15).</summary>
        [TestMethod]
        public void flip()
        {
            var img = cv2.imread(imgSolar);
            var flipped = cv2.flip(img, FLIP_MODE.FLIP_HORIZONTAL_MODE);
            Assert.AreEqual(410, flipped.shape[0]);
            Assert.AreEqual(730, flipped.shape[1]);
            Assert.AreEqual(3, flipped.shape[2]);

            NDArray src = img;
            var vertical = cv2.flip(img, FLIP_MODE.FLIP_VERTICAL_MODE);
            var both = cv2.flip(img, FLIP_MODE.FLIP_BOTH_MODE);
            Oracle.Run(() =>
            {
                Oracle.AssertSame(Oracle.cv2.flip(src, 1), flipped.data, "flip horizontal");
                Oracle.AssertSame(Oracle.cv2.flip(src, 0), vertical.data, "flip vertical");
                Oracle.AssertSame(Oracle.cv2.flip(src, -1), both.data, "flip both");
            });
        }

        /// <summary>An .npy image loaded with NumSharp becomes a Mat without a copy, and back.</summary>
        [TestMethod]
        public void imshow_from_ndarray()
        {
            var mnist_image_5 = (NDArray)np.load(MNIST_Image_5);
            mnist_image_5 = mnist_image_5.reshape((28, 28));
            //cv2.imshow("Show MNIST 5", mnist_image_5);
            //cv2.waitKey(0);
            using var mat = new Mat(mnist_image_5, MatCodecMode.View);
            Assert.AreEqual(MatType.CV_64FC1, mat.MatType);
            Assert.IsTrue(np.array_equal(mnist_image_5, mat.data));
        }
    }
}
