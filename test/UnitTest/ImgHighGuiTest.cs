using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>The data an image viewer reads from a Mat (no window is opened).</summary>
    [TestClass]
    public class ImgHighGuiTest : Test
    {
        /// <summary>The shape and bytes a viewer would display.</summary>
        [TestMethod]
        public void imshow()
        {
            var img = cv2.imread(imgSolar);
			// cv2.imshow("solar", img);
			// cv2.waitKey(0);
			var dv = GetData(img);
            Assert.AreEqual(3, dv.Item1);
            CollectionAssert.AreEqual(new long[] { 410, 730, 3 }, dv.Item2);
            Assert.AreEqual(897900, dv.Item3.Length);
        }

        /// <summary>The rank, dimensions and bytes of a Mat (what a viewer needs).</summary>
        /// <param name="target">A Mat; anything else gives an empty result.</param>
        /// <returns>(rank, dimensions, bytes).</returns>
		public (byte,long[],byte[]) GetData(object target)
		{
			byte size = 0;
			long[] dims = new long[0];
			byte[] data = new byte[0];

			if (target is SharpCV.Mat mat)
			{
				dims = mat.shape.Dimensions;
				size = (byte)dims.Length;
				data = mat.data.ToArray<byte>();
			}

			return (size, dims, data);

		}

	}
}
