using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpCV;
using System;
using System.Collections.Generic;
using System.Text;
using NumSharp;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>Mats made from NDArrays: values, shapes and the view relationship.</summary>
    [TestClass]
    public class CreationTest : Test
    {
        /// <summary>A uint8 2-D array becomes a CV_8UC1 Mat with the same rows.</summary>
        [TestMethod]
        public void ndarray_2dim_byte()
        {
            NDArray kernel = new byte[,]
            {
                { 172, 172, 173 },
                { 174, 174, 174 },
                { 175, 176, 173 }
            };

            var mat = new Mat(kernel);

            Assert.AreEqual((3, 3), mat.shape);
            Assert.AreEqual(kernel[0], mat.data[0]);
            Assert.AreEqual(kernel[1], mat.data[1]);
            Assert.AreEqual(kernel[2], mat.data[2]);
            Assert.AreEqual(MatType.CV_8UC1, mat.MatType);
        }

        /// <summary>A float32 2-D array becomes a CV_32FC1 Mat with the same rows.</summary>
        [TestMethod]
        public void ndarray_2dim_float()
        {
            NDArray kernel = new float[,]
            {
                { 0, -1, 0 },
                { -1, 5, -1 },
                { -2, -1, 3 }
            };

            var mat = new Mat(kernel);

            Assert.AreEqual((3, 3), mat.shape);
            Assert.AreEqual(kernel[0], mat.data[0]);
            Assert.AreEqual(kernel[1], mat.data[1]);
            Assert.AreEqual(kernel[2], mat.data[2]);
            Assert.AreEqual(MatType.CV_32FC1, mat.MatType);
        }

        /// <summary>A channel slice of an image (a column-strided view) becomes a Mat: Auto copies it, as cv2 does, and
        ///     the values survive.</summary>
        [TestMethod]
        public void ndarray_mat_3x5()
        {
            var img = cv2.imread(img3x5);
            var nd = img.data[":", ":", "0"];
            var mat = new Mat(nd);

            Assert.AreEqual((5, 3), nd.Shape);
            Assert.AreEqual(nd[0], mat.data[0]);
            Assert.AreEqual(nd[1], mat.data[1]);
            Assert.AreEqual(nd[2], mat.data[2]);
        }
    }
}
