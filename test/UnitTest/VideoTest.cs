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

            // SharpCV against itself, on every OS: the frame decoded into the caller's array is the frame read() decodes
            // into a Mat of its own.
            using (var again = cv2.VideoCapture(mp4Road))
            {
                Mat mat = null;
                for (int i = 0; i <= 5; i++)
                {
                    mat?.Dispose();
                    mat = again.read().Item2;
                }

                Assert.IsTrue(np.array_equal(mat.data, frame5), "frame 5: the NDArray buffer holds what read() decodes");
                mat.Dispose();
            }

            Oracle.Run(() =>
            {
                dynamic cap = Oracle.cv2.VideoCapture(mp4Road);
                NDArray expected = null;
                for (int i = 0; i <= 5; i++)
                    expected = cap.read()[1];
                cap.release();
                if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows))
                {
                    // Both packages ship OpenCV's own opencv_videoio_ffmpeg4130_64.dll: the same decoder, the same bytes.
                    Oracle.AssertSame(expected, frame5, "frame 5");
                    return;
                }

                // Elsewhere each package links its own FFmpeg (OpenCvSharp's .so statically, opencv-python avcodec
                // 62.11 / swscale 9.1): H.264 decodes bit-exactly by the standard, but swscale's YUV -> BGR conversion
                // differs between versions, so the frames are the same picture a few levels apart. Measured on Linux
                // x64 (2026-11-09): mean |diff| 1.42, max 18. A real defect (swapped channels, a wrong frame, a shifted
                // row) puts the mean in the tens.
                using var diff = np.abs(expected.astype(np.int16) - frame5.astype(np.int16));
                int max = (int)np.amax(diff);
                double mean = (double)np.mean(diff);
                Assert.IsTrue(mean <= 3.0 && max <= 64, $"frame 5 against cv2's (another FFmpeg build): max |diff| {max}, mean {mean:F4}");
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
