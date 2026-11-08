using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using SharpCV;
using System.IO;
using static SharpCV.Binding;

namespace UnitTest
{
    /// <summary>Video decoding through OpenCV's FFmpeg plugin.</summary>
    [TestClass]
    public class VideoTest : Test
    {
        /// <summary>All 38 frames of road.mp4 decode, the last one 1280 x 720.</summary>
        [TestMethod]
        public void VideoCaptureFromFile()
        {
            int frameCnt = 0;
            Mat lastFrame = null;
            var vid = cv2.VideoCapture(mp4Road);
            Assert.IsTrue(vid.isOpened());
            var (loaded, frame) = vid.read();
            while (loaded)
            {
                frameCnt++;
                lastFrame = frame;
                (loaded, frame) = vid.read();
                /*if (loaded)
                {
                    cv2.imshow("video frame", frame);
                    cv2.waitKey(0);
                }*/
            }
            Assert.AreEqual(38, frameCnt);
            Assert.AreEqual(720, lastFrame.shape[0]);
            Assert.AreEqual(1280, lastFrame.shape[1]);
            Assert.AreEqual(3, lastFrame.shape[2]);
        }

        /// <summary>
        ///     Reading into one preallocated NumSharp frame buffer: OpenCV decodes every frame straight into the array's
        ///     memory (the same array comes back each time), and the frames equal cv2's.
        /// </summary>
        [TestMethod]
        public void VideoCaptureIntoNDArray()
        {
            using var vid = cv2.VideoCapture(mp4Road);
            var buffer = np.zeros(new Shape(720, 1280, 3), np.uint8);
            int frames = 0;
            NDArray frame5 = null;
            while (true)
            {
                var (ok, frame) = vid.read(buffer);
                if (!ok)
                    break;
                Assert.IsTrue(ReferenceEquals(buffer, frame), "the frame was decoded into the caller's array");
                if (frames++ == 5)
                    frame5 = buffer.copy();
            }

            Assert.AreEqual(38, frames);
            Oracle.Run(() =>
            {
                dynamic cap = Oracle.cv2.VideoCapture(mp4Road);
                NDArray expected = null;
                for (int i = 0; i <= 5; i++)
                    expected = cap.read()[1];
                cap.release();
                Oracle.AssertSame(expected, frame5, "frame 5");
            });
        }

        /// <summary>A camera loop (needs a camera).</summary>
        [Ignore]
        [TestMethod]
        public void VideoCaptureFromCamera()
        {
            var vid = cv2.VideoCapture(0);
            var (loaded, frame) = vid.read();
            while (loaded)
            {
                (loaded, frame) = vid.read();
                cv2.imshow("camera capture", frame);
                cv2.waitKey(100);
            }
        }
    }
}
